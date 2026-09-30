using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace VehicleCountingRefactor;

public static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            var options = AppOptions.Parse(args);

            using var snapshotStore = new SnapshotStore();
            using var frameStore = new FrameStore();
            using var apiServer = new VehicleCountApiServer(options, snapshotStore, frameStore);
            using var app = new VehicleCountApp(options, snapshotStore, frameStore);

            apiServer.Start();
            app.Run();

            return 0;
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return ex.Message.StartsWith("Usage:", StringComparison.Ordinal) ? 0 : 1;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex);
            return 1;
        }
    }
}

static class AppConfig
{
    public const string DefaultModelFileName = "gas_station_yolo11m.onnx";
    public const int ReferenceFrameWidth = 640;
    public const int ReferenceFrameHeight = 384;
    public const int DefaultImageSize = 640;
    // Reading a LIVE stream must never be throttled: consuming slower than real time makes the
    // client fall behind the playlist window until segments expire ("Stream timeout").
    // Throttle inference instead — that is what leaves CPU headroom for decoding.
    public const double DefaultMaxReaderFps = 0.0;
    public const double DefaultMaxProcessFps = 3.0;
    public const double DefaultStaleTrackSeconds = 2.0;
    public const double DefaultStationaryConfirmSeconds = 5.0;
    public const double DefaultWaitingStationaryMaxDisplacementPx = 14.0;
    // Wanju station charge times, measured per vehicle class.
    public const double DefaultCarServiceMinutes = 7.0;
    public const double DefaultHeavyServiceMinutes = 30.0;

    // Class voting. Calling a bus a car is the expensive mistake — it frees a charger slot that is
    // really blocked and swaps a 30-minute charge for a 7-minute one — so promotion to heavy needs
    // only a strong minority, while dropping back to car needs a clear majority.
    public const int DefaultClassHistoryFrames = 20;
    public const double DefaultClassSwitchMajority = 0.6;
    public const double DefaultHeavyPromoteShare = 0.35;

    // Tracking across a car <-> bus disagreement. Cost multiplier for matching a detection to a
    // track of the other bucket; 0 disables it (pre-2026-09-04 behaviour, where the detection
    // started a NEW track instead — which reset the charger timer and starved the class vote).
    public const double DefaultCrossBucketMatchPenalty = 1.5;

    // Charger/queue damping: how long a brief track loss may last before the estimate collapses.
    public const double DefaultChargerTrackGraceSeconds = 8.0;
    public const double DefaultQueueResetDebounceSeconds = 15.0;

    public const double DuplicateIoUThreshold = 0.50;
    public const double DuplicateIntersectionRatioThreshold = 0.70;
    public const double ChargerHeavyOverlapSuppressionThreshold = 0.60;

    public static readonly IReadOnlySet<string> EmptyTrackIdSet =
        new HashSet<string>(StringComparer.Ordinal);

    public static readonly IReadOnlyList<ModelClassDefinition> ModelClasses =
    [
        new ModelClassDefinition(0, "car", false),
        new ModelClassDefinition(1, "bus", true),
        new ModelClassDefinition(2, "truck", true),
    ];

    public static readonly IReadOnlyList<RegionDefinition> CameraRegions =
    [
        new RegionDefinition(
            "Left Charger",
            "left_charger",
            [new Point(140, 60), new Point(200, 60), new Point(200, 130), new Point(140, 130)],
            new Scalar(255, 42, 4)),

        new RegionDefinition(
            "Right Charger 1",
            "right_charger_1",
            [new Point(250, 60), new Point(330, 60), new Point(330, 105), new Point(250, 105)],
            new Scalar(37, 255, 225)),

        new RegionDefinition(
            "Right Charger 2",
            "right_charger_2",
            [new Point(250, 105), new Point(330, 105), new Point(330, 135), new Point(250, 135)],
            new Scalar(37, 255, 225)),

        new RegionDefinition(
            "Waiting Lane",
            "waiting_lane",
            [new Point(20, 135), new Point(395, 135), new Point(395, 360), new Point(20, 360)],
            new Scalar(80, 180, 255)),
    ];

    public static string DescribeSupportedClasses() =>
        "0=car, 1=bus, 2=truck; bus+truck => heavyVehicles";

    public static bool IsChargerRole(string role) =>
        role.Contains("charger", StringComparison.Ordinal);

    public static bool IsRightChargerRole(string role) =>
        role.StartsWith("right_charger", StringComparison.Ordinal);

    public static bool IsCarClass(int classId) => classId == 0;

    public static bool IsBusClass(int classId) => classId == 1;

    public static bool IsHeavyClass(int classId) => classId is 1 or 2;

    public static VehicleBucket GetBucket(int classId)
    {
        if (IsCarClass(classId)) return VehicleBucket.Car;
        if (IsHeavyClass(classId)) return VehicleBucket.Heavy;
        return VehicleBucket.Unknown;
    }

    public static string GetDisplayLabel(int classId) =>
        classId switch
        {
            0 => "car",
            1 => "bus",
            2 => "truck",
            _ => $"cls{classId}",
        };
}

sealed class VehicleCountApp : IDisposable
{
    private readonly AppOptions _options;
    private readonly SnapshotStore _snapshotStore;
    private readonly FrameStore _frameStore;
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = false };
    private readonly Stopwatch _debugFrameClock = Stopwatch.StartNew();

    private LatestFrameReader? _reader;
    private CameraWorker? _worker;
    private string? _lastJson;

    public VehicleCountApp(AppOptions options, SnapshotStore snapshotStore, FrameStore frameStore)
    {
        _options = options;
        _snapshotStore = snapshotStore;
        _frameStore = frameStore;
    }

    public void Run()
    {
        PrintStartupSummary();

        if (!SourceExists(_options.Source))
        {
            throw new FileNotFoundException($"Source path '{_options.Source}' does not exist.");
        }

        _reader = new LatestFrameReader("right", _options.Source, _options.MaxReaderFps);
        _reader.Start();

        if (!_reader.WaitForFirstFrame(TimeSpan.FromSeconds(15)))
        {
            if (!string.IsNullOrWhiteSpace(_reader.FailureMessage))
            {
                throw new InvalidOperationException($"Unable to read initial frame: {_reader.FailureMessage}");
            }

            throw new InvalidOperationException("Unable to read initial frame.");
        }

        var regions = BuildRegions(_reader.FrameSize);

        _worker = new CameraWorker(
            cameraKey: "right",
            reader: _reader,
            modelPath: _options.ModelPath,
            imageSize: _options.ImageSize,
            confidenceThreshold: _options.ConfidenceThreshold,
            nmsThreshold: _options.NmsThreshold,
            regions: regions,
            staleTrackSeconds: _options.StaleTrackSeconds,
            stationaryConfirmSeconds: _options.StationaryConfirmSeconds,
            waitingStationaryMaxDisplacementPx: _options.WaitingStationaryMaxDisplacementPx,
            carServiceMinutes: _options.CarServiceMinutes,
            heavyServiceMinutes: _options.HeavyServiceMinutes,
            classHistoryFrames: _options.ClassHistoryFrames,
            classSwitchMajority: _options.ClassSwitchMajority,
            heavyPromoteShare: _options.HeavyPromoteShare,
            crossBucketMatchPenalty: _options.CrossBucketMatchPenalty,
            chargerTrackGraceSeconds: _options.ChargerTrackGraceSeconds,
            queueResetDebounceSeconds: _options.QueueResetDebounceSeconds,
            maxProcessFps: _options.MaxProcessFps,
            logStats: _options.LogStats,
            renderFrames: _options.View || _options.DebugFrame);

        _worker.Start();

        if (_options.View)
        {
            Cv2.NamedWindow("Camera 2");
        }

        try
        {
            while (true)
            {
                var status = _worker.GetStatus();

                if (!string.IsNullOrWhiteSpace(status.FailureMessage))
                {
                    throw new InvalidOperationException($"Worker failed: {status.FailureMessage}");
                }

                var snapshot = BuildSnapshot(status);
                EmitSnapshot(snapshot);

                if (_options.View || _options.DebugFrame)
                {
                    using var frame = _worker.GetRenderedFrame();
                    if (frame is not null)
                    {
                        DrawSummary(frame, snapshot);

                        if (_options.View)
                        {
                            Cv2.ImShow("Camera 2", frame);
                        }

                        if (_options.DebugFrame && _debugFrameClock.ElapsedMilliseconds >= 500)
                        {
                            _debugFrameClock.Restart();
                            CaptureDebugFrame(frame);
                        }
                    }
                }

                if (status.Ended)
                {
                    break;
                }

                if (_options.View && Cv2.WaitKey(1) == 'q')
                {
                    break;
                }

                Thread.Sleep(10);
            }
        }
        finally
        {
            _worker.Stop();
            _reader.Stop();
        }
    }

    private Snapshot BuildSnapshot(CameraWorkerStatus status)
    {
        return new Snapshot
        {
            CameraId = _options.CameraId,
            WaitingTime = status.WaitingTime,
            TotalVehicles = status.TotalVehicles,
            Cars = status.Cars,
            HeavyVehicles = status.HeavyVehicles,
            BusyChargerSlots = status.BusyChargerSlots,
            RightBusyChargerSlots = status.RightBusyChargerSlots,
            WaitingLaneVehicles = status.WaitingLaneVehicles,
            StationaryWaitingLaneVehicles = status.StationaryWaitingLaneVehicles,
            WaitingVehicles = status.StationaryWaitingLaneVehicles,
            WaitingCars = status.WaitingCars,
            WaitingHeavyVehicles = status.WaitingHeavyVehicles,
            ChargersBusy = status.BusyChargerSlots,
            ChargersTotal = status.ChargerSlotsTotal,
        };
    }

    private void EmitSnapshot(Snapshot snapshot)
    {
        var json = JsonSerializer.Serialize(snapshot, _jsonOptions);
        _snapshotStore.Update(json);

        if (json == _lastJson)
        {
            return;
        }

        Console.WriteLine(json);
        _lastJson = json;
    }

    private void PrintStartupSummary()
    {
        Console.Error.WriteLine("VehicleCountingRefactor started");
        Console.Error.WriteLine($"cameraId      : {_options.CameraId}");
        Console.Error.WriteLine($"model         : {_options.ModelPath}");
        Console.Error.WriteLine($"source        : {_options.Source}");
        Console.Error.WriteLine($"imgsz         : {_options.ImageSize}");
        Console.Error.WriteLine($"reader fps    : {_options.MaxReaderFps:0.0} (0 = unlimited)");
        Console.Error.WriteLine($"process fps   : {_options.MaxProcessFps:0.0} (0 = unlimited)");
        Console.Error.WriteLine($"classes       : {AppConfig.DescribeSupportedClasses()}");
        Console.Error.WriteLine($"conf / nms    : {_options.ConfidenceThreshold:0.00} / {_options.NmsThreshold:0.00}");
        Console.Error.WriteLine($"charge min    : car {_options.CarServiceMinutes:0.0} / heavy {_options.HeavyServiceMinutes:0.0}");
        Console.Error.WriteLine(
            $"class vote    : {_options.ClassHistoryFrames} frames, switch {_options.ClassSwitchMajority:0.00}, " +
            $"heavy promote {_options.HeavyPromoteShare:0.00}, cross-bucket x{_options.CrossBucketMatchPenalty:0.0}");
        Console.Error.WriteLine(
            $"queue damping : charger grace {_options.ChargerTrackGraceSeconds:0.0}s, " +
            $"reset debounce {_options.QueueResetDebounceSeconds:0.0}s");
        Console.Error.WriteLine($"stationary sec: {_options.StationaryConfirmSeconds:0.0}s");
        Console.Error.WriteLine($"stale sec     : {_options.StaleTrackSeconds:0.0}s");
        Console.Error.WriteLine($"view enabled  : {_options.View}");
        Console.Error.WriteLine($"debug frame   : {_options.DebugFrame}");
        Console.Error.WriteLine($"API endpoint  : {DescribeApiEndpoint()}");
        Console.Error.WriteLine("mode          : single camera + latest frame + simple waiting logic");
    }

    // ListenPrefix may be a HttpListener wildcard ("http://+:8080/" or "http://*:8080/"),
    // which System.Uri cannot parse. Only used for the startup log line, so never let it throw.
    private string DescribeApiEndpoint()
    {
        var route = "/" + _options.ApiRoute.TrimStart('/');
        var prefix = _options.ListenPrefix.TrimEnd('/');

        try
        {
            var parsable = prefix
                .Replace("://+:", "://0.0.0.0:")
                .Replace("://*:", "://0.0.0.0:");
            return new Uri(new Uri(parsable + "/"), route.TrimStart('/')).ToString();
        }
        catch (UriFormatException)
        {
            return prefix + route;
        }
    }

    private void CaptureDebugFrame(Mat frame)
    {
        try
        {
            // Cv2.ImEncode returns void in OpenCvSharp 4.13 — check the buffer instead.
            Cv2.ImEncode(".jpg", frame, out var buffer,
                new ImageEncodingParam(ImwriteFlags.JpegQuality, 80));

            if (buffer is { Length: > 0 })
            {
                _frameStore.Update(buffer);
            }
        }
        catch (Exception ex)
        {
            // The debug frame is a diagnostic aid; never let it take the service down.
            Console.Error.WriteLine($"[debug-frame] encode failed: {ex.Message}");
        }
    }

    private static void DrawSummary(Mat frame, Snapshot snapshot)
    {
        var lines = new[]
        {
            $"WaitingTime: {snapshot.WaitingTime}",
            $"totalVehicles: {snapshot.TotalVehicles}",
            $"cars: {snapshot.Cars}",
            $"heavyVehicles: {snapshot.HeavyVehicles}",
            $"busyChargerSlots: {snapshot.BusyChargerSlots}",
            $"rightBusyChargerSlots: {snapshot.RightBusyChargerSlots}",
            $"waitingLaneVehicles: {snapshot.WaitingLaneVehicles}",
            $"stationaryWaiting: {snapshot.StationaryWaitingLaneVehicles}",
            $"waiting car/heavy: {snapshot.WaitingCars}/{snapshot.WaitingHeavyVehicles}",
            $"chargers: {snapshot.ChargersBusy}/{snapshot.ChargersTotal}",
        };

        for (var i = 0; i < lines.Length; i++)
        {
            var y = 28 + i * 18;
            Cv2.PutText(frame, lines[i], new Point(20, y),
                HersheyFonts.HersheySimplex, 0.45, Scalar.Black, 2, LineTypes.AntiAlias);
            Cv2.PutText(frame, lines[i], new Point(20, y),
                HersheyFonts.HersheySimplex, 0.45, Scalar.White, 1, LineTypes.AntiAlias);
        }
    }

    private static List<Region> BuildRegions(Size frameSize)
    {
        return AppConfig.CameraRegions.Select(def =>
        {
            var points = def.Points
                .Select(p => new Point(
                    (int)Math.Round(p.X * frameSize.Width / (double)AppConfig.ReferenceFrameWidth),
                    (int)Math.Round(p.Y * frameSize.Height / (double)AppConfig.ReferenceFrameHeight)))
                .ToArray();

            var minX = points.Min(p => p.X);
            var minY = points.Min(p => p.Y);
            var maxX = points.Max(p => p.X);
            var maxY = points.Max(p => p.Y);

            return new Region
            {
                Name = def.Name,
                Role = def.Role,
                Points = points,
                Bounds = new Rect(minX, minY, Math.Max(1, maxX - minX), Math.Max(1, maxY - minY)),
                Color = def.Color,
                IsCharger = AppConfig.IsChargerRole(def.Role),
                Counts = 0,
            };
        }).ToList();
    }

    private static bool SourceExists(string source) =>
        int.TryParse(source, out _) ||
        source.Contains("://", StringComparison.Ordinal) ||
        File.Exists(source);

    public void Dispose()
    {
        _worker?.Dispose();
        _reader?.Dispose();
        Cv2.DestroyAllWindows();
    }
}

sealed class LatestFrameReader : IDisposable
{
    private readonly string _cameraKey;
    private readonly string _source;
    private readonly bool _streamSource;
    private readonly double _maxFps;
    private readonly object _sync = new();
    private readonly CancellationTokenSource _cancellation = new();

    private Task? _readerTask;
    private VideoCapture? _capture;
    private Mat? _latestFrame;
    private long _latestFrameId;
    private Size? _frameSize;
    private bool _ended;
    private string? _failureMessage;
    private double _lastSuccessAt;
    private double _lastFrameTime;

    public LatestFrameReader(string cameraKey, string source, double maxFps)
    {
        _cameraKey = cameraKey;
        _source = source;
        _streamSource = int.TryParse(source, out _) || source.Contains("://", StringComparison.Ordinal);
        _maxFps = _streamSource ? Math.Max(0.0, maxFps) : 0.0;
    }

    public void Start()
    {
        _readerTask = Task.Run(ReadLoopAsync);
    }

    public bool WaitForFirstFrame(TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;

        while (DateTime.UtcNow < deadline)
        {
            lock (_sync)
            {
                if (_latestFrame is not null) return true;
                if (_ended) return false;
            }

            Thread.Sleep(50);
        }

        return false;
    }

    public (long frameId, Mat? frame) GetLatestFrame()
    {
        lock (_sync)
        {
            return (_latestFrameId, _latestFrame?.Clone());
        }
    }

    public Size FrameSize => _frameSize ?? throw new InvalidOperationException($"{_cameraKey} frame size unavailable.");

    public string? FailureMessage
    {
        get { lock (_sync) return _failureMessage; }
    }

    public bool Ended
    {
        get { lock (_sync) return _ended; }
    }

    public double LastFrameTime
    {
        get { lock (_sync) return _lastFrameTime; }
    }

    private async Task ReadLoopAsync()
    {
        try
        {
            while (!_cancellation.IsCancellationRequested)
            {
                if (_capture is null && !OpenCapture())
                {
                    await Task.Delay(1000, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                ApplyFpsLimit();

                var frame = new Mat();
                var success = _capture!.Read(frame) && !frame.Empty();

                if (success)
                {
                    var frameTime = TimeUtil.MonotonicSeconds();

                    lock (_sync)
                    {
                        _latestFrame?.Dispose();
                        _latestFrame = frame;
                        _latestFrameId++;
                        _frameSize = new Size(frame.Width, frame.Height);
                        _lastFrameTime = frameTime;
                    }

                    _lastSuccessAt = frameTime;
                    continue;
                }

                frame.Dispose();
                ReleaseCapture();

                if (_streamSource)
                {
                    await Task.Delay(1000, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                lock (_sync)
                {
                    _ended = true;
                }

                break;
            }
        }
        catch (OperationCanceledException) when (_cancellation.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            lock (_sync)
            {
                _failureMessage = ex.Message;
                _ended = true;
            }

            Console.Error.WriteLine($"[{_cameraKey}] reader failed: {ex}");
        }
        finally
        {
            ReleaseCapture();

            lock (_sync)
            {
                _ended = true;
            }
        }
    }

    private bool OpenCapture()
    {
        try
        {
            VideoCapture capture = int.TryParse(_source, out var index)
                ? new VideoCapture(index)
                : new VideoCapture(_source);

            try
            {
                capture.Set(VideoCaptureProperties.BufferSize, 1);
            }
            catch
            {
            }

            if (!capture.IsOpened())
            {
                capture.Dispose();
                return false;
            }

            _capture = capture;
            return true;
        }
        catch (Exception ex)
        {
            lock (_sync)
            {
                _failureMessage = ex.Message;
            }

            throw;
        }
    }

    private void ReleaseCapture()
    {
        _capture?.Release();
        _capture?.Dispose();
        _capture = null;
    }

    private void ApplyFpsLimit()
    {
        if (_maxFps <= 0.0)
        {
            return;
        }

        var minInterval = 1.0 / _maxFps;
        var elapsed = TimeUtil.MonotonicSeconds() - _lastSuccessAt;

        if (elapsed < minInterval)
        {
            Thread.Sleep((int)Math.Ceiling((minInterval - elapsed) * 1000));
        }
    }

    public void Stop()
    {
        _cancellation.Cancel();

        try
        {
            _readerTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
        }

        ReleaseCapture();
    }

    public void Dispose()
    {
        Stop();

        lock (_sync)
        {
            _latestFrame?.Dispose();
            _latestFrame = null;
        }

        _cancellation.Dispose();
    }
}

sealed class CameraWorker : IDisposable
{
    private const double CachedResultReuseSeconds = 10.0;
    private const double StaleFrameThresholdSeconds = 2.0;
    // Smoothing was hiding count flicker by dragging the displayed wait time slowly toward the new
    // value — which showed users a wait time that kept RISING. Flicker is now handled at the source
    // (DebounceCounts + the queue confirm/reset debounces), so the displayed value tracks reality.
    private const double WaitingTimeSmoothingAlpha = 1.0;

    // A count must hold this long before it is published. Stops 2 -> 1 -> 2 single-frame flicker
    // from reaching the website.
    private const double CountConfirmSeconds = 3.0;
    private const int ZeroFrameConfirmThreshold = 3;
    private readonly string _cameraKey;
    private readonly LatestFrameReader _reader;
    private readonly YoloOnnxDetector _detector;
    private readonly SimpleTracker _tracker;
    private readonly List<Region> _regions;
    private readonly SimpleCounter _counter;
    private readonly bool _renderFrames;
    private readonly double _maxProcessFps;
    private readonly bool _logStats;
    private double _lastProcessedAt;
    private readonly object _sync = new();
    private readonly CancellationTokenSource _cancellation = new();

    private Task? _workerTask;
    private Mat? _renderedFrame;
    private CameraWorkerStatus _status;
    private CounterResult? _lastValidResult;
    private double _lastValidTime;
    private (int Total, int Cars, int Heavy)? _publishedCounts;
    private (int Total, int Cars, int Heavy)? _candidateCounts;
    private double _candidateSince;
    private double? _smoothedWaitingSeconds;
    private int _zeroFrameCounter = 0;

    public CameraWorker(
        string cameraKey,
        LatestFrameReader reader,
        string modelPath,
        int imageSize,
        float confidenceThreshold,
        float nmsThreshold,
        List<Region> regions,
        double staleTrackSeconds,
        double stationaryConfirmSeconds,
        double waitingStationaryMaxDisplacementPx,
        double carServiceMinutes,
        double heavyServiceMinutes,
        int classHistoryFrames,
        double classSwitchMajority,
        double heavyPromoteShare,
        double crossBucketMatchPenalty,
        double chargerTrackGraceSeconds,
        double queueResetDebounceSeconds,
        double maxProcessFps,
        bool logStats,
        bool renderFrames)
    {
        _cameraKey = cameraKey;
        _reader = reader;
        _detector = new YoloOnnxDetector(modelPath, imageSize, confidenceThreshold, nmsThreshold);
        _tracker = new SimpleTracker(
            cameraKey,
            staleTrackSeconds,
            classHistoryFrames,
            classSwitchMajority,
            heavyPromoteShare,
            crossBucketMatchPenalty);
        _regions = regions;
        _counter = new SimpleCounter(
            carServiceMinutes,
            heavyServiceMinutes,
            chargerTrackGraceSeconds,
            queueResetDebounceSeconds);
        _maxProcessFps = maxProcessFps;
        _logStats = logStats;
        _renderFrames = renderFrames;

        StationaryConfirmSeconds = stationaryConfirmSeconds;
        WaitingStationaryMaxDisplacementPx = waitingStationaryMaxDisplacementPx;
        _status = CameraWorkerStatus.Empty(cameraKey);
    }

    private double StationaryConfirmSeconds { get; }
    private double WaitingStationaryMaxDisplacementPx { get; }

    public void Start()
    {
        _workerTask = Task.Run(RunLoopAsync);
    }

    public void Stop()
    {
        _cancellation.Cancel();

        try
        {
            _workerTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
        }
    }

    public CameraWorkerStatus GetStatus()
    {
        lock (_sync)
        {
            return _status;
        }
    }

    public Mat? GetRenderedFrame()
    {
        lock (_sync)
        {
            return _renderedFrame?.Clone();
        }
    }

    private async Task RunLoopAsync()
    {
        try
        {
            long lastProcessedFrameId = 0;
            var processedFrames = 0;
            var fpsWindowStartedAt = TimeUtil.MonotonicSeconds();
            var processingFps = 0.0;

            // Periodic throughput report — without it, CPU tuning is guesswork.
            var statsLoggedAt = TimeUtil.MonotonicSeconds();
            var detectMsTotal = 0.0;
            var detectSamples = 0;

            while (!_cancellation.IsCancellationRequested)
            {
                var loopNow = TimeUtil.MonotonicSeconds();
                var (frameId, frame) = _reader.GetLatestFrame();

                if (frame is null)
                {
                    ReuseCachedResultIfFresh(loopNow, lastProcessedFrameId, processingFps);

                    if (_reader.Ended)
                    {
                        UpdateStatus(s => s with
                        {
                            Ended = true,
                            FailureMessage = _reader.FailureMessage,
                        });
                        break;
                    }

                    await Task.Delay(10, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                if (frameId <= lastProcessedFrameId)
                {
                    frame.Dispose();
                    ReuseCachedResultIfFresh(loopNow, frameId, processingFps);

                    if (_reader.Ended && frameId == lastProcessedFrameId)
                    {
                        UpdateStatus(s => s with
                        {
                            Ended = true,
                            FailureMessage = _reader.FailureMessage,
                        });
                        break;
                    }

                    await Task.Delay(5, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                if (_maxProcessFps > 0.0)
                {
                    var minInterval = 1.0 / _maxProcessFps;
                    var sinceLastProcess = loopNow - _lastProcessedAt;

                    if (sinceLastProcess < minInterval)
                    {
                        // Skip this frame. The reader keeps consuming the stream at full rate;
                        // we simply do not spend CPU on inference more often than asked.
                        frame.Dispose();
                        ReuseCachedResultIfFresh(loopNow, frameId, processingFps);

                        var waitMs = (int)Math.Ceiling((minInterval - sinceLastProcess) * 1000);
                        await Task.Delay(Math.Max(5, waitMs), _cancellation.Token)
                            .ContinueWith(_ => { });
                        continue;
                    }
                }

                _lastProcessedAt = loopNow;
                lastProcessedFrameId = frameId;

                var now = TimeUtil.MonotonicSeconds();

                var detectStartedAt = TimeUtil.MonotonicSeconds();
                var detections = _detector.Detect(frame);
                detectMsTotal += (TimeUtil.MonotonicSeconds() - detectStartedAt) * 1000.0;
                detectSamples++;

                var tracks = _tracker.Update(detections, now);

                var analysis = CameraAnalyzer.Analyze(
                    tracks,
                    _regions,
                    StationaryConfirmSeconds,
                    WaitingStationaryMaxDisplacementPx);

                var counted = _counter.Calculate(analysis);
                var stableResult = GetStableResult(
                    now,
                    detections.Count,
                    tracks.Count,
                    counted);
                var debouncedResult = DebounceCounts(now, stableResult);
                var displayResult = ApplyWaitingTimeSmoothing(debouncedResult);

                processedFrames++;
                var fpsElapsed = TimeUtil.MonotonicSeconds() - fpsWindowStartedAt;
                if (fpsElapsed >= 1.0)
                {
                    processingFps = processedFrames / Math.Max(fpsElapsed, 1e-6);
                    processedFrames = 0;
                    fpsWindowStartedAt = TimeUtil.MonotonicSeconds();

                    if (_logStats && fpsWindowStartedAt - statsLoggedAt >= 15.0)
                    {
                        var avgDetectMs = detectSamples > 0 ? detectMsTotal / detectSamples : 0.0;
                        Console.Error.WriteLine(
                            $"[stats] processing {processingFps:0.00} fps, inference {avgDetectMs:0} ms/frame");

                        statsLoggedAt = fpsWindowStartedAt;
                        detectMsTotal = 0.0;
                        detectSamples = 0;
                    }
                }

                Mat? rendered = null;
                if (_renderFrames)
                {
                    rendered = frame.Clone();
                    DrawRegionsAndTracks(rendered, tracks);
                }

                frame.Dispose();

                UpdateStatus(_ => new CameraWorkerStatus(
                    CameraKey: _cameraKey,
                    ProcessingFps: processingFps,
                    FrameId: frameId,
                    TotalVehicles: displayResult.TotalVehicles,
                    Cars: displayResult.Cars,
                    BusyChargerSlots: displayResult.BusyChargerSlots,
                    RightBusyChargerSlots: displayResult.RightBusyChargerSlots,
                    HeavyVehicles: displayResult.HeavyVehicles,
                    WaitingLaneVehicles: displayResult.WaitingLaneVehicles,
                    StationaryWaitingLaneVehicles: displayResult.StationaryWaitingLaneVehicles,
                    WaitingTime: displayResult.WaitingTime,
                    Ended: false,
                    FailureMessage: null,
                    WaitingCars: displayResult.WaitingCars,
                    WaitingHeavyVehicles: displayResult.WaitingHeavyVehicles,
                    ChargerSlotsTotal: displayResult.ChargerSlotsTotal));

                lock (_sync)
                {
                    _renderedFrame?.Dispose();
                    _renderedFrame = rendered;
                }
            }
        }
        catch (OperationCanceledException) when (_cancellation.IsCancellationRequested)
        {
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[{_cameraKey}] worker failed: {ex}");

            UpdateStatus(s => s with
            {
                Ended = true,
                FailureMessage = ex.Message,
            });
        }
    }

    private CounterResult GetStableResult(
        double now,
        int detectionCount,
        int trackCount,
        CounterResult counted)
    {
        if (trackCount > 0)
        {
            _zeroFrameCounter = 0;
            _lastValidResult = counted;
            _lastValidTime = now;
            return counted;
        }

        _zeroFrameCounter++;

        if (_zeroFrameCounter >= ZeroFrameConfirmThreshold)
        {
            return counted;
        }

        if (TryGetReusableCachedResult(now, out var cachedResult))
        {
            return cachedResult;
        }

        return counted;
    }

    private CounterResult DebounceCounts(double now, CounterResult result)
    {
        var incoming = (result.TotalVehicles, result.Cars, result.HeavyVehicles);

        if (_publishedCounts is null || incoming == _publishedCounts.Value)
        {
            _publishedCounts = incoming;
            _candidateCounts = null;
            return result;
        }

        if (_candidateCounts is null || incoming != _candidateCounts.Value)
        {
            // A new value appeared. Start its clock; do not publish it yet.
            _candidateCounts = incoming;
            _candidateSince = now;
        }
        else if (now - _candidateSince >= CountConfirmSeconds)
        {
            _publishedCounts = incoming;
            _candidateCounts = null;
            return result;
        }

        var held = _publishedCounts.Value;

        return new CounterResult
        {
            TotalVehicles = held.Total,
            Cars = held.Cars,
            HeavyVehicles = held.Heavy,
            BusyChargerSlots = result.BusyChargerSlots,
            RightBusyChargerSlots = result.RightBusyChargerSlots,
            WaitingLaneVehicles = result.WaitingLaneVehicles,
            StationaryWaitingLaneVehicles = result.StationaryWaitingLaneVehicles,
            WaitingTime = result.WaitingTime,
            WaitingCars = result.WaitingCars,
            WaitingHeavyVehicles = result.WaitingHeavyVehicles,
            ChargerSlotsTotal = result.ChargerSlotsTotal,
        };
    }

    private CounterResult ApplyWaitingTimeSmoothing(CounterResult result)
    {
        var currentWaitingSeconds = ParseDurationSeconds(result.WaitingTime);

        _smoothedWaitingSeconds = _smoothedWaitingSeconds is null
            ? currentWaitingSeconds
            : (_smoothedWaitingSeconds.Value * (1.0 - WaitingTimeSmoothingAlpha)) +
              (currentWaitingSeconds * WaitingTimeSmoothingAlpha);

        return new CounterResult
        {
            TotalVehicles = result.TotalVehicles,
            Cars = result.Cars,
            HeavyVehicles = result.HeavyVehicles,
            BusyChargerSlots = result.BusyChargerSlots,
            RightBusyChargerSlots = result.RightBusyChargerSlots,
            WaitingLaneVehicles = result.WaitingLaneVehicles,
            StationaryWaitingLaneVehicles = result.StationaryWaitingLaneVehicles,
            WaitingTime = TimeUtil.FormatDuration(_smoothedWaitingSeconds.Value),
            WaitingCars = result.WaitingCars,
            WaitingHeavyVehicles = result.WaitingHeavyVehicles,
            ChargerSlotsTotal = result.ChargerSlotsTotal,
        };
    }

    private void ReuseCachedResultIfFresh(double now, long frameId, double processingFps)
    {
        if (!IsFrameStale(now) || !TryGetReusableCachedResult(now, out var cachedResult))
        {
            return;
        }

        UpdateStatus(_ => new CameraWorkerStatus(
            CameraKey: _cameraKey,
            ProcessingFps: processingFps,
            FrameId: _status.FrameId,
            TotalVehicles: cachedResult.TotalVehicles,
            Cars: cachedResult.Cars,
            BusyChargerSlots: cachedResult.BusyChargerSlots,
            RightBusyChargerSlots: cachedResult.RightBusyChargerSlots,
            HeavyVehicles: cachedResult.HeavyVehicles,
            WaitingLaneVehicles: cachedResult.WaitingLaneVehicles,
            StationaryWaitingLaneVehicles: cachedResult.StationaryWaitingLaneVehicles,
            WaitingTime: cachedResult.WaitingTime,
            Ended: false,
            FailureMessage: null,
            WaitingCars: cachedResult.WaitingCars,
            WaitingHeavyVehicles: cachedResult.WaitingHeavyVehicles,
            ChargerSlotsTotal: cachedResult.ChargerSlotsTotal));
    }

    private bool IsFrameStale(double now)
    {
        var lastFrameTime = _reader.LastFrameTime;
        return lastFrameTime <= 0 || now - lastFrameTime > StaleFrameThresholdSeconds;
    }

    private bool TryGetReusableCachedResult(double now, out CounterResult result)
    {
        if (_lastValidResult is not null && now - _lastValidTime <= CachedResultReuseSeconds)
        {
            result = _lastValidResult;
            return true;
        }

        result = null!;
        return false;
    }

    private static double ParseDurationSeconds(string value)
    {
        var parts = value.Split(':', StringSplitOptions.TrimEntries);
        if (parts.Length != 2 ||
            !int.TryParse(parts[0], CultureInfo.InvariantCulture, out var minutes) ||
            !int.TryParse(parts[1], CultureInfo.InvariantCulture, out var seconds))
        {
            return 0;
        }

        return Math.Max(0, minutes * 60 + seconds);
    }

    private void DrawRegionsAndTracks(Mat frame, IEnumerable<TrackedVehicle> tracks)
    {
        foreach (var region in _regions)
        {
            Cv2.Polylines(frame, [region.Points], true, region.Color, 2);

            Cv2.PutText(
                frame,
                $"{region.Name}: {region.Counts}",
                new Point(region.Bounds.Left, Math.Max(20, region.Bounds.Top - 8)),
                HersheyFonts.HersheySimplex,
                0.5,
                Scalar.Yellow,
                1,
                LineTypes.AntiAlias);
        }

        foreach (var track in tracks)
        {
            var label = $"{AppConfig.GetDisplayLabel(track.ClassId)} #{track.TrackId}";
            if (track.IsStationary)
            {
                label += " [S]";
            }

            var color = track.ClassId switch
            {
                0 => new Scalar(80, 220, 80),
                1 => new Scalar(80, 180, 220),
                2 => new Scalar(80, 220, 220),
                _ => new Scalar(180, 180, 180),
            };

            Cv2.Rectangle(frame, ToRect(track.BoundingBox), color, 2);

            Cv2.PutText(
                frame,
                label,
                new Point((int)track.BoundingBox.X, Math.Max(20, (int)track.BoundingBox.Y - 8)),
                HersheyFonts.HersheySimplex,
                0.45,
                Scalar.White,
                1,
                LineTypes.AntiAlias);
        }
    }

    private void UpdateStatus(Func<CameraWorkerStatus, CameraWorkerStatus> update)
    {
        lock (_sync)
        {
            _status = update(_status);
        }
    }

    private static Rect ToRect(Rect2f rect)
    {
        return new Rect(
            (int)Math.Round(rect.X),
            (int)Math.Round(rect.Y),
            Math.Max(1, (int)Math.Round(rect.Width)),
            Math.Max(1, (int)Math.Round(rect.Height)));
    }

    public void Dispose()
    {
        Stop();
        _detector.Dispose();

        lock (_sync)
        {
            _renderedFrame?.Dispose();
            _renderedFrame = null;
        }

        _cancellation.Dispose();
    }
}

class SimpleCounter
{
    private const double QueueIncreaseConfirmSeconds = 2.0;
    private const double QueueIncreaseThresholdSeconds = 1.0;
    private readonly double _carServiceSeconds;
    private readonly double _heavyServiceSeconds;
    private readonly double _chargerTrackGraceSeconds;
    private readonly double _queueResetDebounceSeconds;
    private readonly List<ChargerSlot> _chargers =
    [
        new ChargerSlot(),
        new ChargerSlot(),
    ];
    private double? _queueStartTime;
    private double _queueEstimatedTotalWaitSeconds;
    private double? _queueInactiveSince;
    private PendingQueueChange? _pendingQueueIncrease;
    private QueueSignature? _confirmedQueueSignature;

    public SimpleCounter(
        double carServiceMinutes,
        double heavyServiceMinutes,
        double chargerTrackGraceSeconds,
        double queueResetDebounceSeconds)
    {
        _carServiceSeconds = carServiceMinutes * 60;
        _heavyServiceSeconds = heavyServiceMinutes * 60;
        _chargerTrackGraceSeconds = Math.Max(0.0, chargerTrackGraceSeconds);
        _queueResetDebounceSeconds = Math.Max(0.0, queueResetDebounceSeconds);
    }

    // Charge time depends on what is plugged in, not on an average.
    private double GetServiceSeconds(TrackedVehicle track) =>
        AppConfig.IsHeavyClass(track.ClassId) ? _heavyServiceSeconds : _carServiceSeconds;

    public CounterResult Calculate(CameraAnalysis analysis)
    {
        var now = TimeUtil.MonotonicSeconds();
        var waitingQueue = analysis.StationaryWaitingLaneVehicles;

        var rightTracks = GetRightChargerTracks(analysis);
        UpdateChargerSlots(now, rightTracks);

        var busySlots = _chargers.Count(slot => slot.TrackId is not null);
        var waitingCars = waitingQueue.Count;
        var queueActive = busySlots == _chargers.Count;
        var waitingSeconds = queueActive
            ? UpdateActiveQueueCountdown(now, waitingQueue)
            : UpdateInactiveQueueCountdown(now);

        return new CounterResult
        {
            TotalVehicles = analysis.RegionVehicles.Count,
            Cars = analysis.RegionVehicles.Count(track => AppConfig.IsCarClass(track.ClassId)),
            HeavyVehicles = analysis.RegionVehicles.Count(track => AppConfig.IsHeavyClass(track.ClassId)),
            BusyChargerSlots = busySlots,
            RightBusyChargerSlots = busySlots,
            WaitingLaneVehicles = analysis.WaitingLaneVehicles.Count,
            StationaryWaitingLaneVehicles = waitingCars,
            WaitingTime = TimeUtil.FormatDuration(waitingSeconds),
            WaitingCars = waitingQueue.Count(track => AppConfig.IsCarClass(track.ClassId)),
            WaitingHeavyVehicles = waitingQueue.Count(track => AppConfig.IsHeavyClass(track.ClassId)),
            ChargerSlotsTotal = _chargers.Count,
        };
    }

    private double UpdateActiveQueueCountdown(double now, List<TrackedVehicle> waitingQueue)
    {
        var simulatedChargers = _chargers
            .Select(slot => slot.Clone())
            .ToList();
        var estimatedTotalWaitSeconds = SimulateQueue(now, simulatedChargers, waitingQueue);
        var signature = BuildQueueSignature(waitingQueue);

        _queueInactiveSince = null;

        if (_queueStartTime is null)
        {
            StartQueueCountdown(now, estimatedTotalWaitSeconds, signature);
            return estimatedTotalWaitSeconds;
        }

        if (_confirmedQueueSignature == signature)
        {
            _pendingQueueIncrease = null;
            return GetRemainingQueueSeconds(now);
        }

        var remainingSeconds = GetRemainingQueueSeconds(now);
        if (estimatedTotalWaitSeconds > remainingSeconds + QueueIncreaseThresholdSeconds)
        {
            TryConfirmQueueIncrease(now, signature, estimatedTotalWaitSeconds);
        }
        else
        {
            _pendingQueueIncrease = null;
        }

        return GetRemainingQueueSeconds(now);
    }

    private double UpdateInactiveQueueCountdown(double now)
    {
        if (_queueStartTime is null)
        {
            return 0;
        }

        _queueInactiveSince ??= now;
        _pendingQueueIncrease = null;

        if (now - _queueInactiveSince.Value >= _queueResetDebounceSeconds)
        {
            ResetQueueCountdown();
            return 0;
        }

        return GetRemainingQueueSeconds(now);
    }

    private void TryConfirmQueueIncrease(double now, QueueSignature signature, double estimatedTotalWaitSeconds)
    {
        if (_pendingQueueIncrease is not null && _pendingQueueIncrease.Value.Signature == signature)
        {
            if (now - _pendingQueueIncrease.Value.FirstSeenAt >= QueueIncreaseConfirmSeconds)
            {
                StartQueueCountdown(now, estimatedTotalWaitSeconds, signature);
                _pendingQueueIncrease = null;
            }

            return;
        }

        _pendingQueueIncrease = new PendingQueueChange(signature, now);
    }

    private void StartQueueCountdown(double now, double estimatedTotalWaitSeconds, QueueSignature signature)
    {
        _queueStartTime = now;
        _queueEstimatedTotalWaitSeconds = Math.Max(0.0, estimatedTotalWaitSeconds);
        _confirmedQueueSignature = signature;
        _queueInactiveSince = null;
        _pendingQueueIncrease = null;
    }

    private double GetRemainingQueueSeconds(double now)
    {
        if (_queueStartTime is null)
        {
            return 0;
        }

        return Math.Max(0.0, _queueEstimatedTotalWaitSeconds - (now - _queueStartTime.Value));
    }

    private void ResetQueueCountdown()
    {
        _queueStartTime = null;
        _queueEstimatedTotalWaitSeconds = 0;
        _queueInactiveSince = null;
        _pendingQueueIncrease = null;
        _confirmedQueueSignature = null;
    }

    private List<TrackedVehicle> GetRightChargerTracks(CameraAnalysis analysis)
    {
        var rightTrackIds = new HashSet<string>(StringComparer.Ordinal);
        var rightTracks = new List<TrackedVehicle>();
        if (analysis.RegionTracks.TryGetValue("right_charger_1", out var rightCharger1Tracks))
        {
            foreach (var track in rightCharger1Tracks)
            {
                if (rightTrackIds.Add(track.ScopedTrackId))
                {
                    rightTracks.Add(track);
                }
            }
        }

        if (analysis.RegionTracks.TryGetValue("right_charger_2", out var rightCharger2Tracks))
        {
            foreach (var track in rightCharger2Tracks)
            {
                if (rightTrackIds.Add(track.ScopedTrackId))
                {
                    rightTracks.Add(track);
                }
            }
        }

        return rightTracks;
    }

    private void UpdateChargerSlots(double now, List<TrackedVehicle> rightTracks)
    {
        var activeTracksById = rightTracks
            .GroupBy(track => track.ScopedTrackId, StringComparer.Ordinal)
            .ToDictionary(group => group.Key, group => group.First(), StringComparer.Ordinal);

        ReleaseExpiredAssignments(now, activeTracksById.Keys);

        // Refresh existing assignments first, then place any newly seen vehicles.
        foreach (var track in activeTracksById.Values
                     .OrderByDescending(track => _chargers.Any(slot => slot.TrackId == track.ScopedTrackId))
                     .ThenByDescending(GetRequiredSlotCount)
                     .ThenBy(track => track.History.Count > 0 ? track.History[0].Timestamp : now))
        {
            EnsureTrackAssignment(now, track);
        }
    }

    // Queue simulation is stateless: it works on a cloned charger snapshot for the current frame only.
    private double SimulateQueue(double now, List<ChargerSlot> chargers, List<TrackedVehicle> queue)
    {
        foreach (var slot in chargers)
        {
            if (slot.TrackId is null || slot.BusyUntil <= now)
            {
                slot.BusyUntil = now;
            }
        }

        var orderedQueue = queue
            .OrderBy(track => track.History.Count > 0 ? track.History[0].Timestamp : now)
            .ToList();

        foreach (var track in orderedQueue)
        {
            var requiredSlots = GetRequiredSlotCount(track);
            var selectedSlots = chargers
                .OrderBy(slot => slot.BusyUntil)
                .Take(requiredSlots)
                .ToList();

            if (selectedSlots.Count < requiredSlots)
            {
                break;
            }

            // FIFO is preserved by scheduling each waiting vehicle in arrival order.
            var startTime = selectedSlots.Max(slot => slot.BusyUntil);
            var endTime = Math.Max(now, startTime) + GetServiceSeconds(track);

            foreach (var slot in selectedSlots)
            {
                slot.BusyUntil = endTime;
            }
        }

        var nextAvailable = chargers.Min(slot => slot.BusyUntil);
        return Math.Max(0.0, nextAvailable - now);
    }

    private void ReleaseExpiredAssignments(double now, IEnumerable<string> activeTrackIds)
    {
        var activeTrackIdSet = new HashSet<string>(activeTrackIds, StringComparer.Ordinal);

        foreach (var group in _chargers
                     .Where(slot => slot.TrackId is not null)
                     .GroupBy(slot => slot.TrackId!, StringComparer.Ordinal)
                     .ToList())
        {
            if (activeTrackIdSet.Contains(group.Key))
            {
                continue;
            }

            var lastSeenAt = group.Max(slot => slot.LastSeenAt);
            if (now - lastSeenAt <= _chargerTrackGraceSeconds)
            {
                continue;
            }

            foreach (var slot in group)
            {
                ClearSlot(slot);
            }
        }
    }

    private void EnsureTrackAssignment(double now, TrackedVehicle track)
    {
        var requiredSlots = GetRequiredSlotCount(track);
        var assignedSlots = _chargers
            .Where(slot => slot.TrackId == track.ScopedTrackId)
            .ToList();

        while (assignedSlots.Count > requiredSlots)
        {
            var slot = assignedSlots[^1];
            ClearSlot(slot);
            assignedSlots.RemoveAt(assignedSlots.Count - 1);
        }

        var serviceSeconds = GetServiceSeconds(track);

        var busyUntil = assignedSlots.Count > 0
            ? assignedSlots.Max(slot => slot.BusyUntil)
            : now + serviceSeconds;

        if (busyUntil <= now)
        {
            busyUntil = now + serviceSeconds;
        }

        var missingSlots = requiredSlots - assignedSlots.Count;
        if (missingSlots > 0)
        {
            var freeSlots = _chargers
                .Where(slot => slot.TrackId is null)
                .Take(missingSlots)
                .ToList();

            foreach (var slot in freeSlots)
            {
                slot.TrackId = track.ScopedTrackId;
                slot.BusyUntil = busyUntil;
                slot.LastSeenAt = now;
                assignedSlots.Add(slot);
            }
        }

        foreach (var slot in assignedSlots)
        {
            slot.TrackId = track.ScopedTrackId;
            slot.BusyUntil = busyUntil;
            slot.LastSeenAt = now;
        }
    }

    private static int GetRequiredSlotCount(TrackedVehicle track) =>
        AppConfig.IsHeavyClass(track.ClassId) ? 2 : 1;

    private static QueueSignature BuildQueueSignature(IEnumerable<TrackedVehicle> waitingQueue)
    {
        var queue = waitingQueue.ToList();
        return new QueueSignature(
            WaitingVehicleCount: queue.Count,
            WaitingDemandSlots: queue.Sum(GetRequiredSlotCount),
            HeavyWaitingCount: queue.Count(track => AppConfig.IsHeavyClass(track.ClassId)));
    }

    private static void ClearSlot(ChargerSlot slot)
    {
        slot.TrackId = null;
        slot.BusyUntil = 0;
        slot.LastSeenAt = 0;
    }

    private readonly record struct QueueSignature(
        int WaitingVehicleCount,
        int WaitingDemandSlots,
        int HeavyWaitingCount);

    private readonly record struct PendingQueueChange(
        QueueSignature Signature,
        double FirstSeenAt);

    private sealed class ChargerSlot
    {
        public string? TrackId { get; set; }
        public double BusyUntil { get; set; }
        public double LastSeenAt { get; set; }

        public ChargerSlot Clone()
        {
            return new ChargerSlot
            {
                TrackId = TrackId,
                BusyUntil = BusyUntil,
                LastSeenAt = LastSeenAt,
            };
        }
    }
}

sealed record CameraWorkerStatus(
    string CameraKey,
    double ProcessingFps,
    long FrameId,
    int TotalVehicles,
    int Cars,
    int BusyChargerSlots,
    int RightBusyChargerSlots,
    int HeavyVehicles,
    int WaitingLaneVehicles,
    int StationaryWaitingLaneVehicles,
    string WaitingTime,
    bool Ended,
    string? FailureMessage,
    int WaitingCars = 0,
    int WaitingHeavyVehicles = 0,
    int ChargerSlotsTotal = 0)
{
    public static CameraWorkerStatus Empty(string cameraKey) =>
        new(
            CameraKey: cameraKey,
            ProcessingFps: 0.0,
            FrameId: 0,
            TotalVehicles: 0,
            Cars: 0,
            BusyChargerSlots: 0,
            RightBusyChargerSlots: 0,
            HeavyVehicles: 0,
            WaitingLaneVehicles: 0,
            StationaryWaitingLaneVehicles: 0,
            WaitingTime: "00:00",
            Ended: false,
            FailureMessage: null);
}

static class CameraAnalyzer
{
    public static CameraAnalysis Analyze(
        List<TrackedVehicle> tracks,
        List<Region> regions,
        double stationaryConfirmSeconds,
        double waitingStationaryMaxDisplacementPx)
    {
        foreach (var region in regions)
        {
            region.Counts = 0;
        }

        var regionVehiclesById = new Dictionary<string, TrackedVehicle>(StringComparer.Ordinal);
        var regionTracks = regions.ToDictionary(
            r => r.Role,
            _ => new List<TrackedVehicle>(),
            StringComparer.Ordinal);
        var waitingLaneVehicles = new List<TrackedVehicle>();
        var stationaryWaitingLaneVehicles = new List<TrackedVehicle>();

        var regionVehicleCounts = regions.ToDictionary(r => r.Role, _ => 0, StringComparer.Ordinal);
        var candidateTracksByRegion = regions.ToDictionary(
            r => r.Role,
            _ => new List<TrackedVehicle>(),
            StringComparer.Ordinal);

        foreach (var track in tracks)
        {
            track.IsStationary = IsStationaryTrack(
                track.History,
                stationaryConfirmSeconds,
                waitingStationaryMaxDisplacementPx);

            foreach (var region in regions)
            {
                if (!ContainsPoint(region, track.Center))
                {
                    continue;
                }

                candidateTracksByRegion[region.Role].Add(track);
            }
        }

        foreach (var region in regions)
        {
            var filtered = FilterRegionTracks(region, candidateTracksByRegion[region.Role]);
            region.Counts = filtered.Count;
            regionVehicleCounts[region.Role] = filtered.Count;
            regionTracks[region.Role].AddRange(filtered);

            foreach (var track in filtered)
            {
                regionVehiclesById.TryAdd(track.ScopedTrackId, track);

                if (region.Role == "waiting_lane")
                {
                    waitingLaneVehicles.Add(track);
                    if (track.IsStationary)
                    {
                        stationaryWaitingLaneVehicles.Add(track);
                    }
                }
            }
        }

        return new CameraAnalysis
        {
            RegionVehicles = regionVehiclesById.Values.ToList(),
            WaitingLaneVehicles = waitingLaneVehicles,
            StationaryWaitingLaneVehicles = stationaryWaitingLaneVehicles,
            RegionTracks = regionTracks,
            RegionVehicleCounts = regionVehicleCounts,
        };
    }

    private static IReadOnlyList<TrackedVehicle> FilterRegionTracks(
        Region region,
        IReadOnlyList<TrackedVehicle> candidateTracks)
    {
        if (!AppConfig.IsChargerRole(region.Role) || candidateTracks.Count < 2)
        {
            return candidateTracks;
        }

        var heavyTracks = candidateTracks.Where(t => AppConfig.IsHeavyClass(t.ClassId)).ToArray();
        if (heavyTracks.Length == 0)
        {
            return candidateTracks;
        }

        return candidateTracks
            .Where(track => !ShouldSuppressCarInsideHeavy(track, heavyTracks))
            .ToArray();
    }

    private static bool ShouldSuppressCarInsideHeavy(
        TrackedVehicle track,
        IReadOnlyList<TrackedVehicle> heavyTracks)
    {
        if (!AppConfig.IsCarClass(track.ClassId))
        {
            return false;
        }

        foreach (var heavyTrack in heavyTracks)
        {
            if (heavyTrack.ScopedTrackId == track.ScopedTrackId)
            {
                continue;
            }

            if (ContainsPoint(heavyTrack.BoundingBox, track.Center))
            {
                return true;
            }

            if (IntersectionOverSmaller(track.BoundingBox, heavyTrack.BoundingBox) >=
                AppConfig.ChargerHeavyOverlapSuppressionThreshold)
            {
                return true;
            }
        }

        return false;
    }

    private static bool ContainsPoint(Region region, Point2f point)
    {
        if (point.X < region.Bounds.Left ||
            point.X > region.Bounds.Right ||
            point.Y < region.Bounds.Top ||
            point.Y > region.Bounds.Bottom)
        {
            return false;
        }

        var contour = region.Points.Select(v => new Point2f(v.X, v.Y)).ToArray();
        return Cv2.PointPolygonTest(contour, point, false) >= 0;
    }

    private static bool ContainsPoint(Rect2f rect, Point2f point)
    {
        return point.X >= rect.Left &&
               point.X <= rect.Right &&
               point.Y >= rect.Top &&
               point.Y <= rect.Bottom;
    }

    private static float IntersectionOverSmaller(Rect2f a, Rect2f b)
    {
        var x1 = Math.Max(a.Left, b.Left);
        var y1 = Math.Max(a.Top, b.Top);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);
        var width = Math.Max(0, x2 - x1);
        var height = Math.Max(0, y2 - y1);
        var intersection = width * height;
        var smaller = Math.Min(a.Width * a.Height, b.Width * b.Height);
        return smaller <= 0 ? 0 : intersection / smaller;
    }

    private static bool IsStationaryTrack(
        List<TrackSample> history,
        double stationaryConfirmSeconds,
        double waitingStationaryMaxDisplacementPx)
    {
        if (history.Count < 2 || stationaryConfirmSeconds <= 0)
        {
            return false;
        }

        var latestTimestamp = history[^1].Timestamp;
        var windowStart = latestTimestamp - stationaryConfirmSeconds;
        var recent = history.Where(s => s.Timestamp >= windowStart).ToArray();

        if (recent.Length < 2)
        {
            return false;
        }

        var observedDuration = recent[^1].Timestamp - recent[0].Timestamp;
        if (observedDuration < stationaryConfirmSeconds * 0.8)
        {
            return false;
        }

        var minX = recent.Min(s => s.Point.X);
        var maxX = recent.Max(s => s.Point.X);
        var minY = recent.Min(s => s.Point.Y);
        var maxY = recent.Max(s => s.Point.Y);

        var spreadX = maxX - minX;
        var spreadY = maxY - minY;
        var maxObservedDisplacement = Math.Sqrt(spreadX * spreadX + spreadY * spreadY);

        return maxObservedDisplacement <= waitingStationaryMaxDisplacementPx;
    }
}

sealed class SnapshotStore : IDisposable
{
    private readonly object _sync = new();
    private string? _latestJson;

    public void Update(string json)
    {
        lock (_sync)
        {
            _latestJson = json;
        }
    }

    public bool TryGetLatestJson(out string json)
    {
        lock (_sync)
        {
            if (_latestJson is null)
            {
                json = string.Empty;
                return false;
            }

            json = _latestJson;
            return true;
        }
    }

    public void Dispose()
    {
    }
}

sealed class FrameStore : IDisposable
{
    private readonly object _sync = new();
    private byte[]? _jpeg;

    public void Update(byte[] jpeg)
    {
        lock (_sync)
        {
            _jpeg = jpeg;
        }
    }

    public bool TryGetLatestJpeg(out byte[] jpeg)
    {
        lock (_sync)
        {
            if (_jpeg is null)
            {
                jpeg = Array.Empty<byte>();
                return false;
            }

            jpeg = _jpeg;
            return true;
        }
    }

    public void Dispose()
    {
    }
}

sealed class VehicleCountApiServer : IDisposable
{
    private readonly AppOptions _options;
    private readonly SnapshotStore _snapshotStore;
    private readonly FrameStore _frameStore;
    private readonly HttpListener _listener = new();
    private readonly CancellationTokenSource _cancellation = new();
    private Task? _serverTask;

    public VehicleCountApiServer(AppOptions options, SnapshotStore snapshotStore, FrameStore frameStore)
    {
        _options = options;
        _snapshotStore = snapshotStore;
        _frameStore = frameStore;
        _listener.Prefixes.Add(NormalizePrefix(options.ListenPrefix));
    }

    public void Start()
    {
        _listener.Start();
        _serverTask = Task.Run(() => RunAsync(_cancellation.Token));
    }

    private async Task RunAsync(CancellationToken cancellationToken)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            HttpListenerContext? context = null;

            try
            {
                context = await _listener.GetContextAsync();
            }
            catch (HttpListenerException)
            {
                break;
            }
            catch (ObjectDisposedException)
            {
                break;
            }

            if (context is not null)
            {
                _ = Task.Run(() => HandleRequestAsync(context), cancellationToken);
            }
        }
    }

    private async Task HandleRequestAsync(HttpListenerContext context)
    {
        try
        {
            var requestPath = context.Request.Url?.AbsolutePath?.TrimEnd('/') ?? string.Empty;
            var apiPath = NormalizePath(_options.ApiRoute);

            if (context.Request.HttpMethod == "GET" && requestPath == apiPath)
            {
                if (_snapshotStore.TryGetLatestJson(out var json))
                {
                    await WriteJsonAsync(context.Response, 200, json);
                }
                else
                {
                    await WriteJsonAsync(context.Response, 503, """{"message":"No detection result available yet."}""");
                }

                return;
            }

            if (context.Request.HttpMethod == "GET" && requestPath == "/health")
            {
                await WriteJsonAsync(context.Response, 200, """{"status":"ok"}""");
                return;
            }

            if (context.Request.HttpMethod == "GET" &&
                (requestPath == "/frame.jpg" || requestPath == "/frame"))
            {
                if (_frameStore.TryGetLatestJpeg(out var jpeg))
                {
                    await WriteBinaryAsync(context.Response, 200, "image/jpeg", jpeg);
                }
                else
                {
                    await WriteJsonAsync(context.Response, 503,
                        """{"message":"No debug frame yet. Start the app with --debug-frame."}""");
                }

                return;
            }

            await WriteJsonAsync(context.Response, 404, """{"message":"Not found"}""");
        }
        catch
        {
            if (context.Response.OutputStream.CanWrite)
            {
                await WriteJsonAsync(context.Response, 500, """{"message":"Internal server error"}""");
            }
        }
    }

    private static async Task WriteJsonAsync(HttpListenerResponse response, int statusCode, string json)
    {
        var payload = Encoding.UTF8.GetBytes(json);
        response.StatusCode = statusCode;
        response.ContentType = "application/json";
        response.ContentEncoding = Encoding.UTF8;
        response.ContentLength64 = payload.LongLength;
        response.Headers["Cache-Control"] = "no-store";
        await response.OutputStream.WriteAsync(payload);
        response.Close();
    }

    private static async Task WriteBinaryAsync(
        HttpListenerResponse response, int statusCode, string contentType, byte[] payload)
    {
        response.StatusCode = statusCode;
        response.ContentType = contentType;
        response.ContentLength64 = payload.LongLength;
        response.Headers["Cache-Control"] = "no-store";
        await response.OutputStream.WriteAsync(payload);
        response.Close();
    }

    private static string NormalizePrefix(string prefix) =>
        prefix.EndsWith("/", StringComparison.Ordinal) ? prefix : $"{prefix}/";

    private static string NormalizePath(string route)
    {
        if (string.IsNullOrWhiteSpace(route))
        {
            return "/vehicle-count";
        }

        var normalized = route.StartsWith("/", StringComparison.Ordinal) ? route : $"/{route}";
        return normalized.TrimEnd('/');
    }

    public void Dispose()
    {
        _cancellation.Cancel();

        if (_listener.IsListening)
        {
            _listener.Stop();
        }

        _listener.Close();

        try
        {
            _serverTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
        }

        _cancellation.Dispose();
    }
}

sealed class YoloOnnxDetector : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private readonly int _classCount;
    private readonly float _confidenceThreshold;
    private readonly float _nmsThreshold;

    public YoloOnnxDetector(
        string modelPath,
        int imageSize,
        float confidenceThreshold,
        float nmsThreshold)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"Model file '{modelPath}' does not exist.");
        }

        _session = new InferenceSession(modelPath);
        _inputName = _session.InputMetadata.Keys.First();

        var inputMetadata = _session.InputMetadata[_inputName];
        _inputWidth = ResolveInputDimension(inputMetadata.Dimensions, 3, imageSize);
        _inputHeight = ResolveInputDimension(inputMetadata.Dimensions, 2, imageSize);

        _classCount = ResolveClassCount(_session);
        _confidenceThreshold = confidenceThreshold;
        _nmsThreshold = nmsThreshold;
    }

    public List<Detection> Detect(Mat frame)
    {
        using var resized = Letterbox(frame, out var scale, out var padX, out var padY);
        using var rgb = new Mat();
        using var floatMat = new Mat();

        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
        rgb.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

        var area = _inputWidth * _inputHeight;
        var inputData = new float[3 * area];

        Cv2.Split(floatMat, out var channels);
        try
        {
            for (var c = 0; c < channels.Length; c++)
            {
                channels[c].GetArray(out float[] channelData);
                Array.Copy(channelData, 0, inputData, c * area, area);
            }
        }
        finally
        {
            foreach (var ch in channels)
            {
                ch.Dispose();
            }
        }

        var tensor = new DenseTensor<float>(inputData, new[] { 1, 3, _inputHeight, _inputWidth });
        using var results = _session.Run([NamedOnnxValue.CreateFromTensor(_inputName, tensor)]);

        var output = results.First().AsTensor<float>();
        return ParseDetections(output, scale, padX, padY, frame.Width, frame.Height);
    }

    private static int ResolveInputDimension(IReadOnlyList<int> dimensions, int axisIndex, int fallback)
    {
        if (axisIndex < dimensions.Count && dimensions[axisIndex] > 0)
        {
            return dimensions[axisIndex];
        }

        return fallback;
    }

    private static int ResolveClassCount(InferenceSession session)
    {
        var outputMetadata = session.OutputMetadata.Values.First();
        var dims = outputMetadata.Dimensions.ToArray();

        if (dims.Length != 3)
        {
            throw new InvalidOperationException("Unsupported YOLO ONNX output metadata.");
        }

        var featureCount = dims[1] > 0 && dims[1] <= dims[2] ? dims[1] : dims[2];
        var classCount = featureCount - 4;

        if (classCount != AppConfig.ModelClasses.Count)
        {
            throw new InvalidOperationException(
                $"Model output exposes {classCount} classes, but this app expects {AppConfig.ModelClasses.Count} classes.");
        }

        return classCount;
    }

    private List<Detection> ParseDetections(
        Tensor<float> output,
        float scale,
        int padX,
        int padY,
        int originalWidth,
        int originalHeight)
    {
        var dimensions = output.Dimensions.ToArray();
        if (dimensions.Length != 3)
        {
            throw new InvalidOperationException("Unsupported YOLO output shape.");
        }

        var featureFirst = dimensions[1] <= dimensions[2];
        var featureCount = featureFirst ? dimensions[1] : dimensions[2];
        var detectionCount = featureFirst ? dimensions[2] : dimensions[1];

        if (featureCount != _classCount + 4)
        {
            throw new InvalidOperationException(
                $"Unexpected YOLO output feature count {featureCount}. Expected {_classCount + 4}.");
        }

        var values = output.ToArray();

        float Read(int featureIndex, int detectionIndex)
        {
            return featureFirst
                ? values[featureIndex * detectionCount + detectionIndex]
                : values[detectionIndex * featureCount + featureIndex];
        }

        var detections = new List<Detection>();

        for (var detectionIndex = 0; detectionIndex < detectionCount; detectionIndex++)
        {
            var bestClass = -1;
            var bestScore = 0f;

            foreach (var modelClass in AppConfig.ModelClasses)
            {
                var score = Read(4 + modelClass.Id, detectionIndex);
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = modelClass.Id;
                }
            }

            if (bestClass < 0 || bestScore < _confidenceThreshold)
            {
                continue;
            }

            var cx = Read(0, detectionIndex);
            var cy = Read(1, detectionIndex);
            var w = Read(2, detectionIndex);
            var h = Read(3, detectionIndex);

            var x1 = Math.Clamp((cx - w / 2f - padX) / scale, 0, originalWidth - 1);
            var y1 = Math.Clamp((cy - h / 2f - padY) / scale, 0, originalHeight - 1);
            var x2 = Math.Clamp((cx + w / 2f - padX) / scale, 0, originalWidth - 1);
            var y2 = Math.Clamp((cy + h / 2f - padY) / scale, 0, originalHeight - 1);

            if (x2 <= x1 || y2 <= y1)
            {
                continue;
            }

            detections.Add(new Detection
            {
                ClassId = bestClass,
                Confidence = bestScore,
                BoundingBox = new Rect2f(x1, y1, x2 - x1, y2 - y1),
                Center = new Point2f((x1 + x2) / 2f, (y1 + y2) / 2f),
            });
        }

        return ApplyDuplicateSuppression(detections);
    }

    private Mat Letterbox(Mat frame, out float scale, out int padX, out int padY)
    {
        scale = Math.Min(_inputWidth / (float)frame.Width, _inputHeight / (float)frame.Height);

        var resizedWidth = Math.Max(1, (int)Math.Round(frame.Width * scale));
        var resizedHeight = Math.Max(1, (int)Math.Round(frame.Height * scale));

        padX = (_inputWidth - resizedWidth) / 2;
        padY = (_inputHeight - resizedHeight) / 2;

        var canvas = new Mat(new Size(_inputWidth, _inputHeight), MatType.CV_8UC3, new Scalar(114, 114, 114));
        using var resized = new Mat();
        Cv2.Resize(frame, resized, new Size(resizedWidth, resizedHeight));
        using var roi = new Mat(canvas, new Rect(padX, padY, resizedWidth, resizedHeight));
        resized.CopyTo(roi);

        return canvas;
    }

    private List<Detection> ApplyDuplicateSuppression(IEnumerable<Detection> detections)
    {
        var kept = new List<Detection>();

        foreach (var detection in detections.OrderByDescending(d => d.Confidence))
        {
            if (kept.Any(existing =>
                    (AppConfig.GetBucket(existing.ClassId) == AppConfig.GetBucket(detection.ClassId) &&
                     IoU(existing.BoundingBox, detection.BoundingBox) >= Math.Max(_nmsThreshold, AppConfig.DuplicateIoUThreshold))
                    ||
                    (AppConfig.GetBucket(existing.ClassId) == AppConfig.GetBucket(detection.ClassId) &&
                     IntersectionOverSmaller(existing.BoundingBox, detection.BoundingBox) >= AppConfig.DuplicateIntersectionRatioThreshold)))
            {
                continue;
            }

            kept.Add(detection);
        }

        return kept;
    }

    private static float IoU(Rect2f a, Rect2f b)
    {
        var x1 = Math.Max(a.Left, b.Left);
        var y1 = Math.Max(a.Top, b.Top);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);
        var width = Math.Max(0, x2 - x1);
        var height = Math.Max(0, y2 - y1);
        var intersection = width * height;
        var union = a.Width * a.Height + b.Width * b.Height - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    private static float IntersectionOverSmaller(Rect2f a, Rect2f b)
    {
        var x1 = Math.Max(a.Left, b.Left);
        var y1 = Math.Max(a.Top, b.Top);
        var x2 = Math.Min(a.Right, b.Right);
        var y2 = Math.Min(a.Bottom, b.Bottom);
        var width = Math.Max(0, x2 - x1);
        var height = Math.Max(0, y2 - y1);
        var intersection = width * height;
        var smaller = Math.Min(a.Width * a.Height, b.Width * b.Height);
        return smaller <= 0 ? 0 : intersection / smaller;
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}

sealed class SimpleTracker
{
    // A vehicle's class does not change from frame to frame — only the detector's opinion does.
    // Vote over a window, and require a clear majority before switching, so a 50/50 car/bus
    // split cannot flip the reported class on every frame.
    private readonly string _cameraKey;
    private readonly Dictionary<int, TrackedVehicle> _tracks = new();
    private readonly double _staleTrackSeconds;
    private readonly int _classHistoryLimit;
    private readonly double _classSwitchMajority;
    private readonly double _heavyPromoteShare;
    private readonly double _crossBucketMatchPenalty;
    private int _nextTrackId = 1;

    public SimpleTracker(
        string cameraKey,
        double staleTrackSeconds,
        int classHistoryFrames,
        double classSwitchMajority,
        double heavyPromoteShare,
        double crossBucketMatchPenalty)
    {
        _cameraKey = cameraKey;
        _staleTrackSeconds = Math.Max(0.1, staleTrackSeconds);
        _classHistoryLimit = Math.Max(1, classHistoryFrames);
        _classSwitchMajority = Math.Clamp(classSwitchMajority, 0.0, 1.0);
        _heavyPromoteShare = Math.Clamp(heavyPromoteShare, 0.0, 1.0);
        _crossBucketMatchPenalty = Math.Max(0.0, crossBucketMatchPenalty);
    }

    public List<TrackedVehicle> Update(List<Detection> detections, double nowSeconds)
    {
        foreach (var track in _tracks.Values)
        {
            track.MissedFrames++;
            track.SeenThisFrame = false;
        }

        var assignments = BuildAssignments(detections);

        foreach (var assignment in assignments)
        {
            var track = _tracks[assignment.TrackId];
            var detection = detections[assignment.DetectionIndex];

            track.ClassId = detection.ClassId;
            track.BoundingBox = detection.BoundingBox;
            track.Center = detection.Center;
            track.MissedFrames = 0;
            track.LastSeenAt = nowSeconds;
            track.SeenThisFrame = true;
            track.History.Add(new TrackSample(detection.Center, nowSeconds));

            if (track.History.Count > 30)
            {
                track.History.RemoveAt(0);
            }

            track.ClassHistory.Add(detection.ClassId);
            if (track.ClassHistory.Count > _classHistoryLimit)
            {
                track.ClassHistory.RemoveAt(0);
            }

            track.ClassId = ResolveStableClass(track);
        }

        var matchedDetectionIndexes = assignments
            .Select(a => a.DetectionIndex)
            .ToHashSet();

        for (var i = 0; i < detections.Count; i++)
        {
            if (matchedDetectionIndexes.Contains(i))
            {
                continue;
            }

            var detection = detections[i];

            _tracks[_nextTrackId] = new TrackedVehicle
            {
                TrackId = _nextTrackId,
                CameraKey = _cameraKey,
                ClassId = detection.ClassId,
                BoundingBox = detection.BoundingBox,
                Center = detection.Center,
                MissedFrames = 0,
                LastSeenAt = nowSeconds,
                SeenThisFrame = true,
                IsStationary = false,
                History = [new TrackSample(detection.Center, nowSeconds)],
                ClassHistory = [detection.ClassId],
            };

            _nextTrackId++;
        }

        foreach (var staleTrackId in _tracks
                     .Where(pair => nowSeconds - pair.Value.LastSeenAt > _staleTrackSeconds)
                     .Select(pair => pair.Key)
                     .ToList())
        {
            _tracks.Remove(staleTrackId);
        }

        return _tracks.Values
            .Where(t => t.SeenThisFrame)
            .Select(t => t.Clone())
            .ToList();
    }

    private List<Assignment> BuildAssignments(IReadOnlyList<Detection> detections)
    {
        var candidates = new List<Assignment>();

        foreach (var track in _tracks.Values)
        {
            for (var i = 0; i < detections.Count; i++)
            {
                var detection = detections[i];

                var sameBucket = BucketsCompatible(track.ClassId, detection.ClassId);

                // A car <-> bus disagreement is the detector changing its mind about ONE vehicle,
                // not a second vehicle appearing. Refusing the match here spawned a NEW track every
                // time the label flipped: the counts jumped, the charger timer restarted, and the
                // class vote never saw the opposing votes it needs. Allow the match, but let a
                // same-bucket candidate win whenever one is in range.
                if (!sameBucket && _crossBucketMatchPenalty <= 0.0)
                {
                    continue;
                }

                var distance = Distance(track.Center, detection.Center);
                var maxDistance = Math.Clamp(
                    Math.Sqrt(track.BoundingBox.Width * track.BoundingBox.Height) * 1.4,
                    45.0,
                    160.0);

                if (distance <= maxDistance)
                {
                    var cost = sameBucket ? distance : distance * _crossBucketMatchPenalty;
                    candidates.Add(new Assignment(track.TrackId, i, cost));
                }
            }
        }

        var assignments = new List<Assignment>();
        var usedTrackIds = new HashSet<int>();
        var usedDetectionIndexes = new HashSet<int>();

        foreach (var candidate in candidates.OrderBy(c => c.Distance))
        {
            if (usedTrackIds.Contains(candidate.TrackId) ||
                usedDetectionIndexes.Contains(candidate.DetectionIndex))
            {
                continue;
            }

            usedTrackIds.Add(candidate.TrackId);
            usedDetectionIndexes.Add(candidate.DetectionIndex);
            assignments.Add(candidate);
        }

        return assignments;
    }

    private int ResolveStableClass(TrackedVehicle track)
    {
        // Deterministic tie-break, and on a tie the heavier class wins: a bus mislabelled as a car
        // frees a charger slot that is really blocked, which is the worse of the two errors.
        var challenger = track.ClassHistory
            .GroupBy(classId => classId)
            .OrderByDescending(group => group.Count())
            .ThenByDescending(group => group.Key)
            .First();

        if (challenger.Key == track.ClassId)
        {
            return track.ClassId;
        }

        var currentCount = track.ClassHistory.Count(classId => classId == track.ClassId);

        // Asymmetric on purpose: promoting to heavy needs only a strong minority, demoting back to
        // car needs a clear majority. A symmetric majority let a bus that was seen as a car early
        // stay a car for as long as it sat there.
        var promotingToHeavy =
            AppConfig.IsHeavyClass(challenger.Key) && !AppConfig.IsHeavyClass(track.ClassId);

        var share = promotingToHeavy ? _heavyPromoteShare : _classSwitchMajority;

        if (challenger.Count() < track.ClassHistory.Count * share)
        {
            return track.ClassId;
        }

        return promotingToHeavy || challenger.Count() > currentCount
            ? challenger.Key
            : track.ClassId;
    }

    private static double Distance(Point2f a, Point2f b)
    {
        var dx = a.X - b.X;
        var dy = a.Y - b.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static bool BucketsCompatible(int firstClassId, int secondClassId)
    {
        var firstBucket = AppConfig.GetBucket(firstClassId);
        var secondBucket = AppConfig.GetBucket(secondClassId);

        return firstBucket != VehicleBucket.Unknown && firstBucket == secondBucket;
    }

    private readonly record struct Assignment(int TrackId, int DetectionIndex, double Distance);
}

sealed class AppOptions
{
    public required string Source { get; init; }
    public required string ModelPath { get; init; }
    public required string CameraId { get; init; }
    public required string ListenPrefix { get; init; }
    public required string ApiRoute { get; init; }
    public required bool View { get; init; }

    public required bool DebugFrame { get; init; }
    public required int ImageSize { get; init; }
    public required double MaxReaderFps { get; init; }

    public required double MaxProcessFps { get; init; }

    public required bool LogStats { get; init; }
    public required float ConfidenceThreshold { get; init; }
    public required float NmsThreshold { get; init; }
    public required double CarServiceMinutes { get; init; }

    public required double HeavyServiceMinutes { get; init; }

    public required int ClassHistoryFrames { get; init; }

    public required double ClassSwitchMajority { get; init; }

    public required double HeavyPromoteShare { get; init; }

    public required double CrossBucketMatchPenalty { get; init; }

    public required double ChargerTrackGraceSeconds { get; init; }

    public required double QueueResetDebounceSeconds { get; init; }
    public required double StaleTrackSeconds { get; init; }
    public required double StationaryConfirmSeconds { get; init; }
    public required double WaitingStationaryMaxDisplacementPx { get; init; }

    public static AppOptions Parse(string[] args)
    {
        string? source = null;
        string? modelPath = null;

        var cameraId = "cam2";
        var listenPrefix = "http://localhost:8080/";
        var apiRoute = "/vehicle-count";
        var view = false;
        var debugFrame = false;
        var imageSize = AppConfig.DefaultImageSize;
        var maxReaderFps = AppConfig.DefaultMaxReaderFps;
        var maxProcessFps = AppConfig.DefaultMaxProcessFps;
        var logStats = false;
        var confidenceThreshold = 0.25f;
        var nmsThreshold = 0.45f;
        var carServiceMinutes = AppConfig.DefaultCarServiceMinutes;
        var heavyServiceMinutes = AppConfig.DefaultHeavyServiceMinutes;
        var classHistoryFrames = AppConfig.DefaultClassHistoryFrames;
        var classSwitchMajority = AppConfig.DefaultClassSwitchMajority;
        var heavyPromoteShare = AppConfig.DefaultHeavyPromoteShare;
        var crossBucketMatchPenalty = AppConfig.DefaultCrossBucketMatchPenalty;
        var chargerTrackGraceSeconds = AppConfig.DefaultChargerTrackGraceSeconds;
        var queueResetDebounceSeconds = AppConfig.DefaultQueueResetDebounceSeconds;
        var staleTrackSeconds = AppConfig.DefaultStaleTrackSeconds;
        var stationaryConfirmSeconds = AppConfig.DefaultStationaryConfirmSeconds;
        var waitingStationaryMaxDisplacementPx = AppConfig.DefaultWaitingStationaryMaxDisplacementPx;

        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--source":
                case "--right-source":
                    source = ReadValue(args, ref i);
                    break;

                case "--model":
                case "--weights":
                    modelPath = ReadValue(args, ref i);
                    break;

                case "--camera-id":
                    cameraId = ReadValue(args, ref i);
                    break;

                case "--listen-prefix":
                    listenPrefix = ReadValue(args, ref i);
                    break;

                case "--api-route":
                    apiRoute = ReadValue(args, ref i);
                    break;

                case "--imgsz":
                    imageSize = int.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--max-reader-fps":
                    maxReaderFps = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--max-process-fps":
                    maxProcessFps = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--stats":
                    logStats = true;
                    break;

                case "--conf":
                    confidenceThreshold = float.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--nms":
                    nmsThreshold = float.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--car-service-minutes":
                    carServiceMinutes = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--heavy-service-minutes":
                case "--bus-service-minutes":
                    heavyServiceMinutes = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                // Legacy flag: one value for every class. Kept so older commands still run.
                case "--service-minutes-per-vehicle":
                    carServiceMinutes = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    heavyServiceMinutes = carServiceMinutes;
                    break;

                case "--class-history-frames":
                    classHistoryFrames = int.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--class-switch-majority":
                    classSwitchMajority = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--heavy-promote-share":
                    heavyPromoteShare = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--cross-bucket-match-penalty":
                    crossBucketMatchPenalty = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--charger-track-grace-seconds":
                    chargerTrackGraceSeconds = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--queue-reset-debounce-seconds":
                    queueResetDebounceSeconds = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--stale-track-seconds":
                    staleTrackSeconds = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--stationary-confirm-seconds":
                    stationaryConfirmSeconds = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--waiting-stationary-max-displacement-px":
                    waitingStationaryMaxDisplacementPx = double.Parse(ReadValue(args, ref i), CultureInfo.InvariantCulture);
                    break;

                case "--view":
                case "--view-img":
                    view = true;
                    break;

                case "--debug-frame":
                    debugFrame = true;
                    break;

                case "--help":
                case "-h":
                    throw new ArgumentException(BuildHelpText());

                default:
                    throw new ArgumentException($"Unknown argument '{args[i]}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(source))
        {
            throw new ArgumentException("A source is required. Use --source.");
        }

        modelPath ??= ResolveDefaultModelPath();

        return new AppOptions
        {
            Source = source,
            ModelPath = modelPath,
            CameraId = cameraId,
            ListenPrefix = listenPrefix,
            ApiRoute = apiRoute,
            View = view,
            DebugFrame = debugFrame,
            ImageSize = imageSize,
            MaxReaderFps = Math.Max(0.0, maxReaderFps),
            MaxProcessFps = Math.Max(0.0, maxProcessFps),
            LogStats = logStats,
            ConfidenceThreshold = confidenceThreshold,
            NmsThreshold = nmsThreshold,
            CarServiceMinutes = Math.Max(0.0, carServiceMinutes),
            HeavyServiceMinutes = Math.Max(0.0, heavyServiceMinutes),
            ClassHistoryFrames = Math.Max(1, classHistoryFrames),
            ClassSwitchMajority = Math.Clamp(classSwitchMajority, 0.0, 1.0),
            HeavyPromoteShare = Math.Clamp(heavyPromoteShare, 0.0, 1.0),
            CrossBucketMatchPenalty = Math.Max(0.0, crossBucketMatchPenalty),
            ChargerTrackGraceSeconds = Math.Max(0.0, chargerTrackGraceSeconds),
            QueueResetDebounceSeconds = Math.Max(0.0, queueResetDebounceSeconds),
            StaleTrackSeconds = Math.Max(0.1, staleTrackSeconds),
            StationaryConfirmSeconds = Math.Max(0.2, stationaryConfirmSeconds),
            WaitingStationaryMaxDisplacementPx = Math.Max(1.0, waitingStationaryMaxDisplacementPx),
        };
    }

    private static string ReadValue(string[] args, ref int index)
    {
        if (index + 1 >= args.Length)
        {
            throw new ArgumentException($"Missing value for '{args[index]}'.");
        }

        index++;
        return args[index];
    }

    private static string ResolveDefaultModelPath()
    {
        var candidates = new[]
        {
            Path.Combine(Environment.CurrentDirectory, AppConfig.DefaultModelFileName),
            Path.Combine(AppContext.BaseDirectory, AppConfig.DefaultModelFileName),
            Path.GetFullPath(Path.Combine(Environment.CurrentDirectory, "..", "..", "TRAIN", "exports", AppConfig.DefaultModelFileName)),
            Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "TRAIN", "exports", AppConfig.DefaultModelFileName)),
        };

        foreach (var candidate in candidates.Distinct(StringComparer.Ordinal))
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return candidates[0];
    }

    private static string BuildHelpText()
    {
        return string.Join(Environment.NewLine,
        [
            "Usage:",
            "  dotnet run -- --source <path|rtsp|index> [options]",
            "",
            "Required:",
            "  --source        Camera 2 source",
            "",
            "Optional:",
            $"  --model         ONNX model path (default: {AppConfig.DefaultModelFileName})",
            "  --camera-id     JSON camera id (default: cam2)",
            "  --listen-prefix HTTP prefix (default: http://localhost:8080/)",
            "  --api-route     HTTP route (default: /vehicle-count)",
            $"  --imgsz         Inference image size (default: {AppConfig.DefaultImageSize})",
            $"  --max-reader-fps Reader FPS cap; 0 = unlimited (default: {AppConfig.DefaultMaxReaderFps:0.0}). Do NOT cap a live stream.",
            $"  --max-process-fps Inference FPS cap; 0 = unlimited (default: {AppConfig.DefaultMaxProcessFps:0.0})",
            "  --stats         Log processing fps and inference time every 15s (off by default)",
            "  --conf          Confidence threshold (default: 0.25)",
            "  --nms           NMS / duplicate suppression threshold (default: 0.45)",
            $"  --car-service-minutes   Charge minutes for a car (default: {AppConfig.DefaultCarServiceMinutes:0.0})",
            $"  --heavy-service-minutes Charge minutes for a bus or truck (default: {AppConfig.DefaultHeavyServiceMinutes:0.0})",
            "  --service-minutes-per-vehicle Legacy: sets both of the above to one value",
            $"  --class-history-frames  Class vote window in frames (default: {AppConfig.DefaultClassHistoryFrames})",
            $"  --class-switch-majority Share of the window needed to change class (default: {AppConfig.DefaultClassSwitchMajority:0.00})",
            $"  --heavy-promote-share   Lower share needed to become bus/truck (default: {AppConfig.DefaultHeavyPromoteShare:0.00})",
            $"  --cross-bucket-match-penalty Cost multiplier for matching a car detection to a heavy track; 0 disables (default: {AppConfig.DefaultCrossBucketMatchPenalty:0.0})",
            $"  --charger-track-grace-seconds Keep a charger slot this long after losing its track (default: {AppConfig.DefaultChargerTrackGraceSeconds:0.0})",
            $"  --queue-reset-debounce-seconds Wait this long before zeroing the queue estimate (default: {AppConfig.DefaultQueueResetDebounceSeconds:0.0})",
            $"  --stale-track-seconds Remove tracks after this many seconds without match (default: {AppConfig.DefaultStaleTrackSeconds:0.0})",
            $"  --stationary-confirm-seconds Seconds vehicle must stay almost still (default: {AppConfig.DefaultStationaryConfirmSeconds:0.0})",
            $"  --waiting-stationary-max-displacement-px Max movement to still count as stationary (default: {AppConfig.DefaultWaitingStationaryMaxDisplacementPx:0.0})",
            "  --view          Show OpenCV preview",
            "  --debug-frame   Serve the annotated frame as JPEG at /frame.jpg (no GUI needed)",
            "",
            $"Supported classes: {AppConfig.DescribeSupportedClasses()}",
        ]);
    }
}

sealed class Snapshot
{
    [JsonPropertyName("cameraId")]
    public required string CameraId { get; init; }

    [JsonPropertyName("WaitingTime")]
    public required string WaitingTime { get; init; }

    [JsonPropertyName("totalVehicles")]
    public required int TotalVehicles { get; init; }

    [JsonPropertyName("cars")]
    public required int Cars { get; init; }

    [JsonPropertyName("heavyVehicles")]
    public required int HeavyVehicles { get; init; }

    // Diagnostics only — shown on the --debug-frame overlay, deliberately NOT in the JSON payload.
    // The API contract stays: cameraId, WaitingTime, totalVehicles, cars, heavyVehicles.
    [JsonIgnore]
    public int WaitingVehicles { get; init; }

    [JsonIgnore]
    public int WaitingCars { get; init; }

    [JsonIgnore]
    public int WaitingHeavyVehicles { get; init; }

    [JsonIgnore]
    public int ChargersBusy { get; init; }

    [JsonIgnore]
    public int ChargersTotal { get; init; }

    [JsonIgnore]
    public int BusyChargerSlots { get; init; }

    [JsonIgnore]
    public int RightBusyChargerSlots { get; init; }

    [JsonIgnore]
    public int WaitingLaneVehicles { get; init; }

    [JsonIgnore]
    public int StationaryWaitingLaneVehicles { get; init; }
}

sealed class TrackedVehicle
{
    public required int TrackId { get; init; }
    public required string CameraKey { get; init; }
    public required int ClassId { get; set; }
    public required Rect2f BoundingBox { get; set; }
    public required Point2f Center { get; set; }
    public required int MissedFrames { get; set; }
    public required double LastSeenAt { get; set; }
    public required bool SeenThisFrame { get; set; }
    public required bool IsStationary { get; set; }
    public required List<TrackSample> History { get; init; }
    public required List<int> ClassHistory { get; init; }

    public string ScopedTrackId => $"{CameraKey}:{TrackId}";

    public TrackedVehicle Clone()
    {
        return new TrackedVehicle
        {
            TrackId = TrackId,
            CameraKey = CameraKey,
            ClassId = ClassId,
            BoundingBox = BoundingBox,
            Center = Center,
            MissedFrames = MissedFrames,
            LastSeenAt = LastSeenAt,
            SeenThisFrame = SeenThisFrame,
            IsStationary = IsStationary,
            History = [.. History],
            ClassHistory = [.. ClassHistory],
        };
    }
}

sealed class Detection
{
    public required int ClassId { get; init; }
    public required float Confidence { get; init; }
    public required Rect2f BoundingBox { get; init; }
    public required Point2f Center { get; init; }
}

sealed class Region
{
    public required string Name { get; init; }
    public required string Role { get; init; }
    public required Point[] Points { get; init; }
    public required Rect Bounds { get; init; }
    public required Scalar Color { get; init; }
    public required bool IsCharger { get; init; }
    public int Counts { get; set; }
}

sealed class CameraAnalysis
{
    public required List<TrackedVehicle> RegionVehicles { get; init; }
    public required List<TrackedVehicle> WaitingLaneVehicles { get; init; }
    public required List<TrackedVehicle> StationaryWaitingLaneVehicles { get; init; }
    public required Dictionary<string, List<TrackedVehicle>> RegionTracks { get; init; }
    public required Dictionary<string, int> RegionVehicleCounts { get; init; }
}

sealed class CounterResult
{
    public required int TotalVehicles { get; init; }
    public required int Cars { get; init; }
    public required int HeavyVehicles { get; init; }
    public required int BusyChargerSlots { get; init; }
    public required int RightBusyChargerSlots { get; init; }
    public required int WaitingLaneVehicles { get; init; }
    public required int StationaryWaitingLaneVehicles { get; init; }
    public required string WaitingTime { get; init; }

    // Waiting-queue breakdown. The station only cares about vehicles that are WAITING —
    // a vehicle occupying a charger is being served, not waiting.
    public int WaitingCars { get; init; }
    public int WaitingHeavyVehicles { get; init; }
    public int ChargerSlotsTotal { get; init; }
}

sealed record RegionDefinition(string Name, string Role, Point[] Points, Scalar Color);
sealed record TrackSample(Point2f Point, double Timestamp);
sealed record ModelClassDefinition(int Id, string Name, bool IsHeavy);

enum VehicleBucket
{
    Unknown = 0,
    Car = 1,
    Heavy = 2,
}

static class TimeUtil
{
    public static double MonotonicSeconds()
    {
        return Stopwatch.GetTimestamp() / (double)Stopwatch.Frequency;
    }

    public static string FormatDuration(double seconds)
    {
        var totalSeconds = Math.Max(0, (int)Math.Floor(seconds));
        var minutes = totalSeconds / 60;
        var remainingSeconds = totalSeconds % 60;
        return $"{minutes:00}:{remainingSeconds:00}";
    }
}
