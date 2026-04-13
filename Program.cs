using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

/*
HOW TO RUN
dotnet run -- \
  --model gas_station_yolo11m.onnx \
  --source cam2.mp4 \
  --camera-id cam2 \
  --listen-prefix http://localhost:8080/ \
  --api-route /vehicle-count \
  --view

JSON OUTPUT
{
  "cameraId": "cam2",
  "WaitingTime": "00:20",
  "totalVehicles": 5,
  "cars": 3,
  "heavyVehicles": 2
}
*/

return AppEntry.Run(args);

static class AppEntry
{
    public static int Run(string[] args)
    {
        try
        {
            var options = AppOptions.Parse(args);
            using var snapshotStore = new SnapshotStore();
            using var apiServer = new VehicleCountApiServer(options, snapshotStore);
            apiServer.Start();
            using var app = new VehicleCountApp(options, snapshotStore);
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
    public const double DefaultMaxReaderFps = 15.0;
    public const int TrackHistoryLength = 30;
    public const int TrackClassHistoryLength = 8;
    public const double DefaultStaleTrackSeconds = 2.0;
    public const double DuplicateIoUThreshold = 0.5;
    public const double DuplicateIntersectionRatioThreshold = 0.7;
    public const double ChargerOccupancyGraceSeconds = 0.5;
    public const double WaitingLaneOccupancyGraceSeconds = 0.5;
    public const double DefaultEstimatedServiceMinutesPerVehicle = 20.0;
    public const double DefaultStationaryConfirmSeconds = 2.5;
    public const double DefaultWaitingStationaryMaxDisplacementPx = 14.0;
    public static readonly IReadOnlySet<string> EmptyTrackIdSet = new HashSet<string>(StringComparer.Ordinal);
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
            [new Point(250, 105), new Point(330, 105), new Point(330, 140), new Point(250, 140)],
            new Scalar(37, 255, 225)),
        new RegionDefinition(
            "Waiting Lane",
            "waiting_lane",
            [new Point(20, 140), new Point(395, 140), new Point(395, 360), new Point(20, 360)],
            new Scalar(80, 180, 255)),
    ];

    public static string DescribeSupportedClasses()
    {
        return "0=car, 1=bus, 2=truck; bus+truck => heavyVehicles";
    }

    public static bool IsChargerRole(string role)
    {
        return role.Contains("charger", StringComparison.Ordinal);
    }

    public static bool IsRightChargerRole(string role)
    {
        return role.StartsWith("right_charger", StringComparison.Ordinal);
    }

    public static bool IsRightCharger1Role(string role)
    {
        return string.Equals(role, "right_charger_1", StringComparison.Ordinal);
    }

    public static bool IsRightCharger2Role(string role)
    {
        return string.Equals(role, "right_charger_2", StringComparison.Ordinal);
    }

    public static bool IsCarClass(int classId)
    {
        return classId == 0;
    }

    public static bool IsHeavyClass(int classId)
    {
        return classId is 1 or 2;
    }

    public static bool IsBusClass(int classId)
    {
        return classId == 1;
    }

    public static VehicleBucket GetBucket(int classId)
    {
        if (IsCarClass(classId))
        {
            return VehicleBucket.Car;
        }

        if (IsHeavyClass(classId))
        {
            return VehicleBucket.Heavy;
        }

        return VehicleBucket.Unknown;
    }

    public static string GetDisplayLabel(int classId)
    {
        return classId switch
        {
            0 => "car",
            1 => "bus",
            2 => "truck",
            _ => $"cls{classId}",
        };
    }
}

sealed class VehicleCountApp : IDisposable
{
    private readonly AppOptions _options;
    private readonly SnapshotStore _snapshotStore;
    private readonly JsonSerializerOptions _jsonOptions = new() { WriteIndented = false };
    private readonly Dictionary<string, LatestFrameReader> _readers = new();
    private readonly Dictionary<string, CameraWorker> _workers = new();
    private string? _lastJson;
    private readonly double _serviceSecondsPerVehicle;

    private double _estimatedWaitRemainingSeconds;
    private double? _estimatedWaitLastUpdatedAt;
    private int _lastObservedAheadVehicleCount;
    private int? _pendingAheadVehicleIncreaseCount;
    private double? _pendingAheadVehicleIncreaseSince;

    public VehicleCountApp(AppOptions options, SnapshotStore snapshotStore)
    {
        _options = options;
        _snapshotStore = snapshotStore;
        _serviceSecondsPerVehicle = Math.Max(0.0, options.ServiceMinutesPerVehicle) * 60.0;
    }

    public void Run()
    {
        PrintStartupSummary();
        StartReaders();

        try
        {
            var frameSizes = WaitForInitialFrames();
            StartWorkers(frameSizes);

            if (_options.View)
            {
                Cv2.NamedWindow("Camera 2");
            }

            while (true)
            {
                var statuses = new Dictionary<string, CameraWorkerStatus>();
                var frames = new Dictionary<string, Mat>();
                var allEnded = true;

                foreach (var pair in _workers)
                {
                    var status = pair.Value.GetStatus();
                    statuses[pair.Key] = status;
                    if (!string.IsNullOrWhiteSpace(status.FailureMessage))
                    {
                        throw new InvalidOperationException($"{pair.Key} worker failed: {status.FailureMessage}");
                    }

                    if (!status.Ended)
                    {
                        allEnded = false;
                    }

                    var frame = pair.Value.GetRenderedFrame();
                    if (frame is not null)
                    {
                        frames[pair.Key] = frame;
                    }
                }

                var snapshot = BuildSnapshot(statuses);
                EmitSnapshot(snapshot);

                if (_options.View)
                {
                    foreach (var pair in frames)
                    {
                        DrawSummary(pair.Value, snapshot, pair.Key);
                    }

                    if (frames.TryGetValue("right", out var rightFrame))
                    {
                        Cv2.ImShow("Camera 2", rightFrame);
                    }
                }

                foreach (var frame in frames.Values)
                {
                    frame.Dispose();
                }

                if (allEnded)
                {
                    break;
                }

                if (_options.View && Cv2.WaitKey(1) == 'q')
                {
                    break;
                }

                if (frames.Count == 0)
                {
                    Thread.Sleep(10);
                }
            }
        }
        finally
        {
            foreach (var worker in _workers.Values)
            {
                worker.Stop();
            }

            foreach (var reader in _readers.Values)
            {
                reader.Stop();
            }
        }
    }

    private void PrintStartupSummary()
    {
        Console.Error.WriteLine("VehicleCountingONNX started");
        Console.Error.WriteLine($"cameraId      : {_options.CameraId}");
        Console.Error.WriteLine($"model         : {_options.ModelPath}");
        Console.Error.WriteLine($"source        : {_options.Source}");
        Console.Error.WriteLine($"imgsz         : {_options.ImageSize}");
        Console.Error.WriteLine($"reader fps    : {_options.MaxReaderFps:0.0}");
        Console.Error.WriteLine($"classes       : {AppConfig.DescribeSupportedClasses()}");
        Console.Error.WriteLine($"conf / nms    : {_options.ConfidenceThreshold:0.00} / {_options.NmsThreshold:0.00}");
        Console.Error.WriteLine($"charger grace : {_options.ChargerOccupancyGraceSeconds:0.0}s");
        Console.Error.WriteLine($"waiting grace : {_options.WaitingLaneOccupancyGraceSeconds:0.0}s");
        Console.Error.WriteLine($"service min   : {_options.ServiceMinutesPerVehicle:0.0}");
        Console.Error.WriteLine($"stationary sec: {_options.StationaryConfirmSeconds:0.0}s");
        Console.Error.WriteLine($"stale sec     : {_options.StaleTrackSeconds:0.0}s");
        Console.Error.WriteLine($"view enabled  : {_options.View}");
        Console.Error.WriteLine($"API endpoint  : {new Uri(new Uri(_options.ListenPrefix), _options.ApiRoute.TrimStart('/'))}");
        Console.Error.WriteLine("mode          : parallel workers + latest frame only");
    }

    private void StartReaders()
    {
        foreach (var pair in new Dictionary<string, string>
                 {
                     ["right"] = _options.Source,
                 })
        {
            if (!SourceExists(pair.Value))
            {
                throw new FileNotFoundException($"Source path '{pair.Value}' does not exist.");
            }

            var reader = new LatestFrameReader(pair.Key, pair.Value, _options.MaxReaderFps);
            reader.Start();
            _readers[pair.Key] = reader;
        }
    }

    private Dictionary<string, Size> WaitForInitialFrames()
    {
        var frameSizes = new Dictionary<string, Size>();
        foreach (var pair in _readers)
        {
            if (!pair.Value.WaitForFirstFrame(TimeSpan.FromSeconds(15)))
            {
                if (!string.IsNullOrWhiteSpace(pair.Value.FailureMessage))
                {
                    throw new InvalidOperationException($"Unable to read an initial frame from {pair.Key} source: {pair.Value.FailureMessage}");
                }

                throw new InvalidOperationException($"Unable to read an initial frame from {pair.Key} source.");
            }

            frameSizes[pair.Key] = pair.Value.FrameSize;
        }

        return frameSizes;
    }

    private void StartWorkers(Dictionary<string, Size> frameSizes)
    {
        foreach (var pair in _readers)
        {
            var regions = BuildRegions(frameSizes[pair.Key]);
            var worker = new CameraWorker(
                pair.Key,
                pair.Value,
                _options.ModelPath,
                _options.ImageSize,
                _options.ConfidenceThreshold,
                _options.NmsThreshold,
                regions,
                _options.ChargerOccupancyGraceSeconds,
                _options.WaitingLaneOccupancyGraceSeconds,
                _options.StaleTrackSeconds,
                _options.StationaryConfirmSeconds,
                _options.WaitingStationaryMaxDisplacementPx,
                _options.View);
            worker.Start();
            _workers[pair.Key] = worker;
        }
    }

    private Snapshot BuildSnapshot(Dictionary<string, CameraWorkerStatus> statuses)
    {
        var rightStatus = statuses.GetValueOrDefault("right") ?? CameraWorkerStatus.Empty("right");

        var totalVehicles = rightStatus.TotalVehicles;
        var cars = rightStatus.Cars;
        var heavyVehicles = rightStatus.HeavyVehicles;
        var waitingTime = UpdateWaitingTimeForQueueEvents(rightStatus);

        return new Snapshot
        {
            CameraId = _options.CameraId,
            WaitingTime = waitingTime,
            TotalVehicles = totalVehicles,
            Cars = cars,
            HeavyVehicles = heavyVehicles,
            BusyChargerSlots = rightStatus.BusyChargerSlots,
            RightBusyChargerSlots = rightStatus.RightBusyChargerSlots,
            WaitingLaneVehicles = rightStatus.WaitingLaneVehicles,
            StationaryWaitingLaneVehicles = rightStatus.StationaryWaitingLaneVehicles,
        };
    }

    private string UpdateWaitingTimeForQueueEvents(CameraWorkerStatus rightStatus)
    {
        var now = TimeUtil.MonotonicSeconds();
        if (_estimatedWaitLastUpdatedAt.HasValue)
        {
            _estimatedWaitRemainingSeconds = Math.Max(0.0, _estimatedWaitRemainingSeconds - (now - _estimatedWaitLastUpdatedAt.Value));
        }
        _estimatedWaitLastUpdatedAt = now;

        var hasStationaryQueue = rightStatus.StationaryWaitingTrackIds.Count > 0;
        var rightBothBusy = rightStatus.RightCharger1Occupied && rightStatus.RightCharger2Occupied;
        var right2BlocksRight1 = rightStatus.RightCharger2Occupied && !rightStatus.RightCharger1Occupied && hasStationaryQueue;
        var right2BusCharging = rightStatus.RightCharger2BusOccupied;
        var right2KeepsActiveEta = rightStatus.RightCharger2Occupied && _estimatedWaitRemainingSeconds > 0.0;
        var shouldRunEta = rightBothBusy || right2BlocksRight1 || right2BusCharging || right2KeepsActiveEta;

        if (!shouldRunEta)
        {
            _estimatedWaitRemainingSeconds = 0.0;
            _lastObservedAheadVehicleCount = 0;
            _pendingAheadVehicleIncreaseCount = null;
            _pendingAheadVehicleIncreaseSince = null;
            return "00:00";
        }

        var currentAheadVehicleTrackIds = new HashSet<string>(rightStatus.StationaryWaitingTrackIds, StringComparer.Ordinal);
        currentAheadVehicleTrackIds.UnionWith(rightStatus.RightCharger1StationaryTrackIds);
        currentAheadVehicleTrackIds.UnionWith(rightStatus.RightCharger2TrackIds);
        var currentAheadVehicleCount = currentAheadVehicleTrackIds.Count;

        if (_estimatedWaitRemainingSeconds <= 0.0)
        {
            _estimatedWaitRemainingSeconds = Math.Max(1, currentAheadVehicleCount) * _serviceSecondsPerVehicle;
            _lastObservedAheadVehicleCount = currentAheadVehicleCount;
            _pendingAheadVehicleIncreaseCount = null;
            _pendingAheadVehicleIncreaseSince = null;
        }
        else if (currentAheadVehicleCount > _lastObservedAheadVehicleCount)
        {
            if (_pendingAheadVehicleIncreaseCount != currentAheadVehicleCount)
            {
                _pendingAheadVehicleIncreaseCount = currentAheadVehicleCount;
                _pendingAheadVehicleIncreaseSince = now;
            }
            else if (_pendingAheadVehicleIncreaseSince.HasValue &&
                     now - _pendingAheadVehicleIncreaseSince.Value >= 1.0)
            {
                _estimatedWaitRemainingSeconds +=
                    (currentAheadVehicleCount - _lastObservedAheadVehicleCount) * _serviceSecondsPerVehicle;
                _lastObservedAheadVehicleCount = currentAheadVehicleCount;
                _pendingAheadVehicleIncreaseCount = null;
                _pendingAheadVehicleIncreaseSince = null;
            }
        }
        else
        {
            _lastObservedAheadVehicleCount = currentAheadVehicleCount;
            _pendingAheadVehicleIncreaseCount = null;
            _pendingAheadVehicleIncreaseSince = null;
        }

        if (_estimatedWaitRemainingSeconds <= 0.0)
        {
            return "00:00";
        }

        return TimeUtil.FormatDuration(_estimatedWaitRemainingSeconds);
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

    private static void DrawSummary(Mat frame, Snapshot snapshot, string cameraKey)
    {
        var summaryLines = new List<string>
        {
            $"WaitingTime: {snapshot.WaitingTime}",
            $"totalVehicles: {snapshot.TotalVehicles}",
            $"cars: {snapshot.Cars}",
            $"heavyVehicles: {snapshot.HeavyVehicles}",
            $"busyChargerSlots: {snapshot.BusyChargerSlots}",
            $"rightBusyChargerSlots: {snapshot.RightBusyChargerSlots}",
            $"waitingLaneVehicles: {snapshot.WaitingLaneVehicles}",
            $"stationaryWaiting: {snapshot.StationaryWaitingLaneVehicles}",
        };

        for (var i = 0; i < summaryLines.Count; i++)
        {
            var y = 28 + i * 18;
            Cv2.PutText(frame, summaryLines[i], new Point(20, y), HersheyFonts.HersheySimplex, 0.45, Scalar.Black, 2, LineTypes.AntiAlias);
            Cv2.PutText(frame, summaryLines[i], new Point(20, y), HersheyFonts.HersheySimplex, 0.45, Scalar.White, 1, LineTypes.AntiAlias);
        }
    }

    private static List<Region> BuildRegions(Size frameSize)
    {
        return AppConfig.CameraRegions.Select(definition =>
        {
            var points = definition.Points
                .Select(point => new Point(
                    (int)Math.Round(point.X * frameSize.Width / (double)AppConfig.ReferenceFrameWidth),
                    (int)Math.Round(point.Y * frameSize.Height / (double)AppConfig.ReferenceFrameHeight)))
                .ToArray();

            var minX = points.Min(point => point.X);
            var minY = points.Min(point => point.Y);
            var maxX = points.Max(point => point.X);
            var maxY = points.Max(point => point.Y);

            return new Region
            {
                Name = definition.Name,
                Role = definition.Role,
                Points = points,
                Bounds = new Rect(minX, minY, Math.Max(1, maxX - minX), Math.Max(1, maxY - minY)),
                Color = definition.Color,
                Counts = 0,
                IsCharger = AppConfig.IsChargerRole(definition.Role),
            };
        }).ToList();
    }

    private static bool SourceExists(string source)
    {
        return int.TryParse(source, out _) || source.Contains("://", StringComparison.Ordinal) || File.Exists(source);
    }

    public void Dispose()
    {
        foreach (var worker in _workers.Values)
        {
            worker.Dispose();
        }

        foreach (var reader in _readers.Values)
        {
            reader.Dispose();
        }

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
    private int _reconnectCount;
    private double _lastSuccessAt;

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
                if (_latestFrame is not null)
                {
                    return true;
                }
                if (_ended)
                {
                    return false;
                }
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

    public Size FrameSize => _frameSize ?? throw new InvalidOperationException($"{_cameraKey} frame size is not available yet.");

    public string? FailureMessage
    {
        get
        {
            lock (_sync)
            {
                return _failureMessage;
            }
        }
    }

    public bool Ended
    {
        get
        {
            lock (_sync)
            {
                return _ended;
            }
        }
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
                    lock (_sync)
                    {
                        _latestFrame?.Dispose();
                        _latestFrame = frame;
                        _latestFrameId += 1;
                        _frameSize = new Size(frame.Width, frame.Height);
                    }
                    _lastSuccessAt = TimeUtil.MonotonicSeconds();
                    continue;
                }

                lock (_sync)
                {
                    frame.Dispose();
                }
                ReleaseCapture();
                if (_streamSource)
                {
                    lock (_sync)
                    {
                        _reconnectCount += 1;
                    }
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
                _ended = true;
                _failureMessage = ex.Message;
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
            VideoCapture capture;
            if (int.TryParse(_source, out var cameraIndex))
            {
                capture = new VideoCapture(cameraIndex);
            }
            else
            {
                capture = new VideoCapture(_source);
            }

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
        if (_maxFps <= 0)
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
    private readonly string _cameraKey;
    private readonly LatestFrameReader _reader;
    private readonly YoloOnnxDetector _detector;
    private readonly SimpleTracker _tracker;
    private readonly List<Region> _regions;
    private readonly Dictionary<string, RegionPresenceState> _regionPresenceStates;
    private readonly bool _renderFrames;
    private readonly object _sync = new();
    private readonly CancellationTokenSource _cancellation = new();
    private Task? _workerTask;
    private Mat? _renderedFrame;
    private CameraWorkerStatus _status = CameraWorkerStatus.Empty("unknown");

    public CameraWorker(
        string cameraKey,
        LatestFrameReader reader,
        string modelPath,
        int imageSize,
        float confidenceThreshold,
        float nmsThreshold,
        List<Region> regions,
        double chargerGraceSeconds,
        double waitingLaneGraceSeconds,
        double staleTrackSeconds,
        double stationaryConfirmSeconds,
        double waitingStationaryMaxDisplacementPx,
        bool renderFrames)
    {
        _cameraKey = cameraKey;
        _reader = reader;
        _detector = new YoloOnnxDetector(modelPath, imageSize, confidenceThreshold, nmsThreshold);
        _tracker = new SimpleTracker(cameraKey, staleTrackSeconds);
        _regions = regions;
        _regionPresenceStates = regions
            .Where(region => region.IsCharger || region.Role == "waiting_lane")
            .ToDictionary(region => region.Role, _ => new RegionPresenceState());
        _renderFrames = renderFrames;
        ChargerGraceSeconds = chargerGraceSeconds;
        WaitingLaneGraceSeconds = waitingLaneGraceSeconds;
        StationaryConfirmSeconds = stationaryConfirmSeconds;
        WaitingStationaryMaxDisplacementPx = waitingStationaryMaxDisplacementPx;
        _status = CameraWorkerStatus.Empty(cameraKey);
    }

    private double ChargerGraceSeconds { get; }
    private double WaitingLaneGraceSeconds { get; }
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
            var chargerRoles = _regions.Where(region => region.IsCharger).Select(region => region.Role).ToArray();

            while (!_cancellation.IsCancellationRequested)
            {
                var (frameId, frame) = _reader.GetLatestFrame();
                if (frame is null)
                {
                    if (_reader.Ended)
                    {
                        var failureMessage = _reader.FailureMessage;
                        UpdateStatus(status => status with
                        {
                            Ended = true,
                            FailureMessage = failureMessage,
                        });
                        break;
                    }

                    await Task.Delay(10, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                if (frameId <= lastProcessedFrameId)
                {
                    frame.Dispose();
                    if (_reader.Ended && frameId == lastProcessedFrameId)
                    {
                        var failureMessage = _reader.FailureMessage;
                        UpdateStatus(status => status with
                        {
                            Ended = true,
                            FailureMessage = failureMessage,
                        });
                        break;
                    }
                    await Task.Delay(5, _cancellation.Token).ContinueWith(_ => { });
                    continue;
                }

                lastProcessedFrameId = frameId;

                var now = TimeUtil.MonotonicSeconds();
                var detections = _detector.Detect(frame);
                var tracks = _tracker.Update(detections, now);
                var analysis = CameraAnalyzer.Analyze(tracks, _regions, StationaryConfirmSeconds, WaitingStationaryMaxDisplacementPx);
                var stationaryWaitingTrackIds = new HashSet<string>(
                    analysis.StationaryWaitingLaneVehicles.Select(track => track.ScopedTrackId),
                    StringComparer.Ordinal);
                var waitingCounts = ApplyRegionOccupancyGrace(
                    _regions,
                    analysis.WaitingLaneVehicles.Count,
                    analysis.StationaryWaitingLaneVehicles.Count,
                    stationaryWaitingTrackIds,
                    analysis.RegionTrackIds,
                    analysis.RegionStationaryTrackIds,
                    analysis.RegionCarCounts,
                    analysis.RegionBusCounts,
                    analysis.RegionHeavyCounts,
                    _regionPresenceStates,
                    ChargerGraceSeconds,
                    WaitingLaneGraceSeconds);
                var retainedChargerVehicles = 0;
                var retainedChargerCars = 0;
                var retainedChargerHeavyVehicles = 0;
                var busyChargerSlots = 0;
                var rightBusyChargerSlots = 0;
                var rightCharger1Occupied = false;
                var rightCharger2Occupied = false;
                var chargerTrackIds = new Dictionary<string, IReadOnlySet<string>>(StringComparer.Ordinal);
                var rightCharger1StationaryTrackIds = AppConfig.EmptyTrackIdSet;
                var rightCharger2BusOccupied = false;

                foreach (var chargerRole in chargerRoles)
                {
                    var chargerCounts = GetEffectiveChargerClassCounts(_regionPresenceStates, chargerRole);
                    chargerTrackIds[chargerRole] = chargerCounts.trackIds;
                    var detectedChargerVehicles = analysis.RegionVehicleCounts.GetValueOrDefault(chargerRole, 0);
                    var detectedChargerCars = analysis.RegionCarCounts.GetValueOrDefault(chargerRole, 0);
                    var detectedChargerBuses = analysis.RegionBusCounts.GetValueOrDefault(chargerRole, 0);
                    var detectedChargerHeavyVehicles = analysis.RegionHeavyCounts.GetValueOrDefault(chargerRole, 0);

                    retainedChargerVehicles += Math.Max(0, chargerCounts.totalVehicles - detectedChargerVehicles);
                    retainedChargerCars += Math.Max(0, chargerCounts.cars - detectedChargerCars);
                    retainedChargerHeavyVehicles += Math.Max(0, chargerCounts.heavyVehicles - detectedChargerHeavyVehicles);
                    if (chargerCounts.totalVehicles > 0)
                    {
                        busyChargerSlots += 1;
                        if (AppConfig.IsRightChargerRole(chargerRole))
                        {
                            rightBusyChargerSlots += 1;
                            if (AppConfig.IsRightCharger1Role(chargerRole))
                            {
                                rightCharger1Occupied = true;
                                rightCharger1StationaryTrackIds = chargerCounts.stationaryTrackIds;
                            }
                            else if (AppConfig.IsRightCharger2Role(chargerRole))
                            {
                                rightCharger2Occupied = true;
                            }

                            if (AppConfig.IsRightCharger2Role(chargerRole) && (chargerCounts.buses > 0 || detectedChargerBuses > 0))
                            {
                                rightCharger2BusOccupied = true;
                            }
                        }
                    }
                }

                processedFrames += 1;
                var fpsElapsed = TimeUtil.MonotonicSeconds() - fpsWindowStartedAt;
                if (fpsElapsed >= 1.0)
                {
                    processingFps = processedFrames / Math.Max(fpsElapsed, 1e-6);
                    processedFrames = 0;
                    fpsWindowStartedAt = TimeUtil.MonotonicSeconds();
                }

                Mat? rendered = null;
                if (_renderFrames)
                {
                    rendered = frame.Clone();
                    DrawRegionsAndTracks(rendered, tracks);
                }
                frame.Dispose();

                UpdateStatus(_ => new CameraWorkerStatus(
                    _cameraKey,
                    processingFps,
                    frameId,
                    analysis.RegionVehicles.Count + retainedChargerVehicles,
                    analysis.RegionVehicles.Count(track => AppConfig.IsCarClass(track.ClassId)) + retainedChargerCars,
                    busyChargerSlots,
                    rightBusyChargerSlots,
                    rightCharger1Occupied,
                    rightCharger2Occupied,
                    rightCharger2BusOccupied,
                    rightCharger1StationaryTrackIds,
                    chargerTrackIds.GetValueOrDefault("right_charger_2", AppConfig.EmptyTrackIdSet),
                    analysis.RegionVehicles.Count(track => AppConfig.IsHeavyClass(track.ClassId)) + retainedChargerHeavyVehicles,
                    waitingCounts.waitingLaneVehicles,
                    waitingCounts.stationaryWaitingLaneVehicles,
                    waitingCounts.stationaryWaitingTrackIds,
                    false,
                    null));

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
            UpdateStatus(status => status with
            {
                Ended = true,
                FailureMessage = ex.Message,
            });
        }
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
            var label = AppConfig.GetDisplayLabel(track.ClassId);
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
                0.5,
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

    private static (int waitingLaneVehicles, int stationaryWaitingLaneVehicles, IReadOnlySet<string> stationaryWaitingTrackIds) ApplyRegionOccupancyGrace(
        List<Region> regions,
        int waitingLaneVehicles,
        int stationaryWaitingLaneVehicles,
        IReadOnlySet<string> stationaryWaitingTrackIds,
        Dictionary<string, IReadOnlySet<string>> regionTrackIds,
        Dictionary<string, IReadOnlySet<string>> regionStationaryTrackIds,
        Dictionary<string, int> regionCarCounts,
        Dictionary<string, int> regionBusCounts,
        Dictionary<string, int> regionHeavyCounts,
        Dictionary<string, RegionPresenceState> regionPresenceStates,
        double chargerOccupancyGraceSeconds,
        double waitingLaneOccupancyGraceSeconds)
    {
        var now = TimeUtil.MonotonicSeconds();
        foreach (var region in regions.Where(region => region.IsCharger || region.Role == "waiting_lane"))
        {
            var state = regionPresenceStates[region.Role];
            var graceSeconds = region.IsCharger ? chargerOccupancyGraceSeconds : waitingLaneOccupancyGraceSeconds;

            if (region.Counts > 0)
            {
                state.LastSeenAt = now;
                state.LastCount = region.Counts;
                if (region.Role == "waiting_lane")
                {
                    state.LastStationaryCount = stationaryWaitingLaneVehicles;
                    state.LastStationaryTrackIds = new HashSet<string>(stationaryWaitingTrackIds, StringComparer.Ordinal);
                }
                else
                {
                    state.LastTrackIds = regionTrackIds.GetValueOrDefault(region.Role, AppConfig.EmptyTrackIdSet);
                    state.LastStationaryTrackIds = regionStationaryTrackIds.GetValueOrDefault(region.Role, AppConfig.EmptyTrackIdSet);
                    state.LastCarCount = regionCarCounts.GetValueOrDefault(region.Role, 0);
                    state.LastBusCount = regionBusCounts.GetValueOrDefault(region.Role, 0);
                    state.LastHeavyCount = regionHeavyCounts.GetValueOrDefault(region.Role, 0);
                }
                continue;
            }

            if (state.LastCount > 0 && now - state.LastSeenAt <= graceSeconds)
            {
                region.Counts = state.LastCount;
                if (region.Role == "waiting_lane")
                {
                    waitingLaneVehicles = state.LastCount;
                    stationaryWaitingLaneVehicles = state.LastStationaryCount;
                    stationaryWaitingTrackIds = state.LastStationaryTrackIds;
                }
            }
            else
            {
                state.LastCount = 0;
                if (region.Role == "waiting_lane")
                {
                    state.LastStationaryCount = 0;
                    state.LastStationaryTrackIds = AppConfig.EmptyTrackIdSet;
                }
                else
                {
                    state.LastTrackIds = AppConfig.EmptyTrackIdSet;
                    state.LastStationaryTrackIds = AppConfig.EmptyTrackIdSet;
                    state.LastCarCount = 0;
                    state.LastBusCount = 0;
                    state.LastHeavyCount = 0;
                }
            }
        }

        return (waitingLaneVehicles, stationaryWaitingLaneVehicles, stationaryWaitingTrackIds);
    }

    private static (int totalVehicles, int cars, int buses, int heavyVehicles, IReadOnlySet<string> trackIds, IReadOnlySet<string> stationaryTrackIds) GetEffectiveChargerClassCounts(
        Dictionary<string, RegionPresenceState> regionPresenceStates,
        string chargerRole)
    {
        if (!regionPresenceStates.TryGetValue(chargerRole, out var state))
        {
            return (0, 0, 0, 0, AppConfig.EmptyTrackIdSet, AppConfig.EmptyTrackIdSet);
        }

        return (state.LastCount, state.LastCarCount, state.LastBusCount, state.LastHeavyCount, state.LastTrackIds, state.LastStationaryTrackIds);
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

sealed record CameraWorkerStatus(
    string CameraKey,
    double ProcessingFps,
    long FrameId,
    int TotalVehicles,
    int Cars,
    int BusyChargerSlots,
    int RightBusyChargerSlots,
    bool RightCharger1Occupied,
    bool RightCharger2Occupied,
    bool RightCharger2BusOccupied,
    IReadOnlySet<string> RightCharger1StationaryTrackIds,
    IReadOnlySet<string> RightCharger2TrackIds,
    int HeavyVehicles,
    int WaitingLaneVehicles,
    int StationaryWaitingLaneVehicles,
    IReadOnlySet<string> StationaryWaitingTrackIds,
    bool Ended,
    string? FailureMessage)
{
    public static CameraWorkerStatus Empty(string cameraKey)
    {
        return new CameraWorkerStatus(
            cameraKey, 0.0, 0, 0, 0, 0, 0, false, false, false,
            AppConfig.EmptyTrackIdSet, AppConfig.EmptyTrackIdSet, 0, 0, 0,
            new HashSet<string>(StringComparer.Ordinal), false, null);
    }
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

        var regionVehicles = new List<TrackedVehicle>();
        var waitingLaneVehicles = new List<TrackedVehicle>();
        var stationaryWaitingLaneVehicles = new List<TrackedVehicle>();
        var regionTrackIds = regions.ToDictionary(
            region => region.Role,
            _ => (IReadOnlySet<string>)new HashSet<string>(StringComparer.Ordinal),
            StringComparer.Ordinal);
        var regionStationaryTrackIds = regions.ToDictionary(
            region => region.Role,
            _ => (IReadOnlySet<string>)new HashSet<string>(StringComparer.Ordinal),
            StringComparer.Ordinal);
        var regionVehicleCounts = regions.ToDictionary(region => region.Role, _ => 0);
        var regionCarCounts = regions.ToDictionary(region => region.Role, _ => 0);
        var regionBusCounts = regions.ToDictionary(region => region.Role, _ => 0);
        var regionHeavyCounts = regions.ToDictionary(region => region.Role, _ => 0);

        foreach (var track in tracks)
        {
            var insideAnyRegion = false;
            var insideWaitingLane = false;
            var isStationary = IsStationaryTrack(track.History, stationaryConfirmSeconds, waitingStationaryMaxDisplacementPx);

            foreach (var region in regions)
            {
                if (!ContainsPoint(region, track.Center))
                {
                    continue;
                }

                region.Counts += 1;
                ((HashSet<string>)regionTrackIds[region.Role]).Add(track.ScopedTrackId);
                if (isStationary)
                {
                    ((HashSet<string>)regionStationaryTrackIds[region.Role]).Add(track.ScopedTrackId);
                }
                regionVehicleCounts[region.Role] += 1;
                if (AppConfig.IsCarClass(track.ClassId))
                {
                    regionCarCounts[region.Role] += 1;
                }
                else if (AppConfig.IsBusClass(track.ClassId))
                {
                    regionBusCounts[region.Role] += 1;
                    regionHeavyCounts[region.Role] += 1;
                }
                else if (AppConfig.IsHeavyClass(track.ClassId))
                {
                    regionHeavyCounts[region.Role] += 1;
                }
                insideAnyRegion = true;
                if (region.Role == "waiting_lane")
                {
                    insideWaitingLane = true;
                }
            }

            if (insideAnyRegion)
            {
                regionVehicles.Add(track);
            }

            if (insideWaitingLane)
            {
                waitingLaneVehicles.Add(track);
                if (isStationary)
                {
                    stationaryWaitingLaneVehicles.Add(track);
                }
            }
        }

        return new CameraAnalysis
        {
            RegionVehicles = regionVehicles,
            WaitingLaneVehicles = waitingLaneVehicles,
            StationaryWaitingLaneVehicles = stationaryWaitingLaneVehicles,
            RegionTrackIds = regionTrackIds,
            RegionStationaryTrackIds = regionStationaryTrackIds,
            RegionVehicleCounts = regionVehicleCounts,
            RegionCarCounts = regionCarCounts,
            RegionBusCounts = regionBusCounts,
            RegionHeavyCounts = regionHeavyCounts,
        };
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

        var contour = region.Points.Select(vertex => new Point2f(vertex.X, vertex.Y)).ToArray();
        return Cv2.PointPolygonTest(contour, point, false) >= 0;
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
        var recent = history.Where(sample => sample.Timestamp >= windowStart).ToArray();
        if (recent.Length < 2)
        {
            return false;
        }

        var observedDuration = recent[^1].Timestamp - recent[0].Timestamp;
        if (observedDuration < stationaryConfirmSeconds * 0.8)
        {
            return false;
        }

        var minX = recent.Min(sample => sample.Point.X);
        var maxX = recent.Max(sample => sample.Point.X);
        var minY = recent.Min(sample => sample.Point.Y);
        var maxY = recent.Max(sample => sample.Point.Y);
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

sealed class VehicleCountApiServer : IDisposable
{
    private readonly AppOptions _options;
    private readonly SnapshotStore _snapshotStore;
    private readonly HttpListener _listener = new();
    private readonly CancellationTokenSource _cancellation = new();
    private Task? _serverTask;

    public VehicleCountApiServer(AppOptions options, SnapshotStore snapshotStore)
    {
        _options = options;
        _snapshotStore = snapshotStore;
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

    private static string NormalizePrefix(string prefix)
    {
        return prefix.EndsWith("/", StringComparison.Ordinal) ? prefix : $"{prefix}/";
    }

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

    public YoloOnnxDetector(string modelPath, int imageSize, float confidenceThreshold, float nmsThreshold)
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
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);
        using var floatMat = new Mat();
        rgb.ConvertTo(floatMat, MatType.CV_32FC3, 1.0 / 255.0);

        var area = _inputWidth * _inputHeight;
        var inputData = new float[3 * area];
        Cv2.Split(floatMat, out var channels);
        for (var channelIndex = 0; channelIndex < channels.Length; channelIndex++)
        {
            channels[channelIndex].GetArray(out float[] channelData);
            Array.Copy(channelData, 0, inputData, channelIndex * area, area);
            channels[channelIndex].Dispose();
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
        var dims = outputMetadata.Dimensions;
        if (dims.Count() != 3)
        {
            throw new InvalidOperationException("Unsupported YOLO ONNX output metadata.");
        }

        var featureCount = dims[1] > 0 && dims[1] <= dims[2] ? dims[1] : dims[2];
        var classCount = featureCount - 4;
        if (classCount != AppConfig.ModelClasses.Count)
        {
            throw new InvalidOperationException(
                $"Model output exposes {classCount} classes, but this gas-station app expects {AppConfig.ModelClasses.Count} classes " +
                "(car, bus, truck). Export and load the fine-tuned gas-station ONNX model.");
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
                $"Unexpected YOLO output feature count {featureCount}. Expected {_classCount + 4} for the gas-station model.");
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
                var scoreIndex = 4 + modelClass.Id;
                var score = Read(scoreIndex, detectionIndex);
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
        foreach (var detection in detections.OrderByDescending(detection => detection.Confidence))
        {
            if (kept.Any(existing =>
                    AppConfig.GetBucket(existing.ClassId) == AppConfig.GetBucket(detection.ClassId) &&
                    IoU(existing.BoundingBox, detection.BoundingBox) >= AppConfig.DuplicateIoUThreshold ||
                    AppConfig.GetBucket(existing.ClassId) == AppConfig.GetBucket(detection.ClassId) &&
                    IntersectionOverSmaller(existing.BoundingBox, detection.BoundingBox) >= AppConfig.DuplicateIntersectionRatioThreshold))
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
    private readonly string _cameraKey;
    private readonly Dictionary<int, TrackedVehicle> _tracks = new();
    private readonly double _staleTrackSeconds;
    private int _nextTrackId = 1;

    public SimpleTracker(string cameraKey, double staleTrackSeconds)
    {
        _cameraKey = cameraKey;
        _staleTrackSeconds = Math.Max(0.1, staleTrackSeconds);
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
            if (track.History.Count > AppConfig.TrackHistoryLength)
            {
                track.History.RemoveAt(0);
            }

            track.ClassHistory.Add(detection.ClassId);
            if (track.ClassHistory.Count > AppConfig.TrackClassHistoryLength)
            {
                track.ClassHistory.RemoveAt(0);
            }

            track.ClassId = track.ClassHistory
                .GroupBy(classId => classId)
                .OrderByDescending(group => group.Count())
                .ThenByDescending(group => group.Last())
                .First()
                .Key;
        }

        var matchedDetections = assignments.Select(assignment => assignment.DetectionIndex).ToHashSet();
        for (var detectionIndex = 0; detectionIndex < detections.Count; detectionIndex++)
        {
            if (matchedDetections.Contains(detectionIndex))
            {
                continue;
            }

            var detection = detections[detectionIndex];
            var track = new TrackedVehicle
            {
                TrackId = _nextTrackId++,
                CameraKey = _cameraKey,
                ClassId = detection.ClassId,
                BoundingBox = detection.BoundingBox,
                Center = detection.Center,
                MissedFrames = 0,
                LastSeenAt = nowSeconds,
                SeenThisFrame = true,
                History = [new TrackSample(detection.Center, nowSeconds)],
                ClassHistory = [detection.ClassId],
            };

            _tracks[track.TrackId] = track;
        }

        foreach (var staleTrackId in _tracks
                     .Where(pair => nowSeconds - pair.Value.LastSeenAt > _staleTrackSeconds)
                     .Select(pair => pair.Key)
                     .ToList())
        {
            _tracks.Remove(staleTrackId);
        }

        return _tracks.Values
            .Where(track => track.SeenThisFrame)
            .Select(track => track.Clone())
            .ToList();
    }

    private List<Assignment> BuildAssignments(IReadOnlyList<Detection> detections)
    {
        var candidates = new List<Assignment>();
        foreach (var track in _tracks.Values)
        {
            for (var detectionIndex = 0; detectionIndex < detections.Count; detectionIndex++)
            {
                var detection = detections[detectionIndex];
                if (!BucketsCompatible(track.ClassId, detection.ClassId))
                {
                    continue;
                }

                var distance = Distance(track.Center, detection.Center);
                var maxDistance = Math.Clamp(Math.Sqrt(track.BoundingBox.Width * track.BoundingBox.Height) * 1.4, 45.0, 160.0);
                if (distance <= maxDistance)
                {
                    candidates.Add(new Assignment(track.TrackId, detectionIndex, distance));
                }
            }
        }

        var assignments = new List<Assignment>();
        var usedTrackIds = new HashSet<int>();
        var usedDetectionIndexes = new HashSet<int>();
        foreach (var candidate in candidates.OrderBy(candidate => candidate.Distance))
        {
            if (usedTrackIds.Contains(candidate.TrackId) || usedDetectionIndexes.Contains(candidate.DetectionIndex))
            {
                continue;
            }

            usedTrackIds.Add(candidate.TrackId);
            usedDetectionIndexes.Add(candidate.DetectionIndex);
            assignments.Add(candidate);
        }

        return assignments;
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
    public required int ImageSize { get; init; }
    public required double MaxReaderFps { get; init; }
    public required float ConfidenceThreshold { get; init; }
    public required float NmsThreshold { get; init; }
    public required double ChargerOccupancyGraceSeconds { get; init; }
    public required double WaitingLaneOccupancyGraceSeconds { get; init; }
    public required double ServiceMinutesPerVehicle { get; init; }
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
        var imageSize = AppConfig.DefaultImageSize;
        var maxReaderFps = AppConfig.DefaultMaxReaderFps;
        var confidenceThreshold = 0.25f;
        var nmsThreshold = 0.45f;
        var chargerOccupancyGraceSeconds = AppConfig.ChargerOccupancyGraceSeconds;
        var waitingLaneOccupancyGraceSeconds = AppConfig.WaitingLaneOccupancyGraceSeconds;
        var serviceMinutesPerVehicle = AppConfig.DefaultEstimatedServiceMinutesPerVehicle;
        var staleTrackSeconds = AppConfig.DefaultStaleTrackSeconds;
        var stationaryConfirmSeconds = AppConfig.DefaultStationaryConfirmSeconds;
        var waitingStationaryMaxDisplacementPx = AppConfig.DefaultWaitingStationaryMaxDisplacementPx;

        for (var index = 0; index < args.Length; index++)
        {
            switch (args[index])
            {
                case "--right-source":
                case "--source":
                    source = ReadValue(args, ref index);
                    break;
                case "--model":
                case "--weights":
                    modelPath = ReadValue(args, ref index);
                    break;
                case "--camera-id":
                    cameraId = ReadValue(args, ref index);
                    break;
                case "--listen-prefix":
                    listenPrefix = ReadValue(args, ref index);
                    break;
                case "--api-route":
                    apiRoute = ReadValue(args, ref index);
                    break;
                case "--imgsz":
                    imageSize = int.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--max-reader-fps":
                    maxReaderFps = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--conf":
                    confidenceThreshold = float.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--nms":
                    nmsThreshold = float.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--charger-occupancy-grace-seconds":
                    chargerOccupancyGraceSeconds = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--waiting-lane-occupancy-grace-seconds":
                    waitingLaneOccupancyGraceSeconds = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--service-minutes-per-vehicle":
                    serviceMinutesPerVehicle = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--stale-track-seconds":
                    staleTrackSeconds = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--stationary-confirm-seconds":
                    stationaryConfirmSeconds = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--waiting-stationary-max-displacement-px":
                    waitingStationaryMaxDisplacementPx = double.Parse(ReadValue(args, ref index), CultureInfo.InvariantCulture);
                    break;
                case "--view":
                case "--view-img":
                    view = true;
                    break;
                case "--help":
                case "-h":
                    throw new ArgumentException(BuildHelpText());
                default:
                    throw new ArgumentException($"Unknown argument '{args[index]}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(source))
        {
            throw new ArgumentException("A camera 2 source is required. Use --source or --right-source.");
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
            ImageSize = imageSize,
            MaxReaderFps = Math.Max(0.0, maxReaderFps),
            ConfidenceThreshold = confidenceThreshold,
            NmsThreshold = nmsThreshold,
            ChargerOccupancyGraceSeconds = Math.Max(0.0, chargerOccupancyGraceSeconds),
            WaitingLaneOccupancyGraceSeconds = Math.Max(0.0, waitingLaneOccupancyGraceSeconds),
            ServiceMinutesPerVehicle = Math.Max(0.0, serviceMinutesPerVehicle),
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
            "  --right-source  Alias for --source",
            "  --camera-id     JSON camera id (default: cam2)",
            "  --listen-prefix HTTP prefix (default: http://localhost:8080/)",
            "  --api-route     HTTP route (default: /vehicle-count)",
            $"  --imgsz         Inference image size (default: {AppConfig.DefaultImageSize})",
            $"  --max-reader-fps Cap stream reader FPS for latest-frame mode (default: {AppConfig.DefaultMaxReaderFps:0.0})",
            "  --conf          Confidence threshold (default: 0.25)",
            "  --nms           Duplicate suppression threshold (default: 0.45)",
            $"  --charger-occupancy-grace-seconds Grace for charger misses (default: {AppConfig.ChargerOccupancyGraceSeconds:0.0})",
            $"  --waiting-lane-occupancy-grace-seconds Grace for waiting-lane misses (default: {AppConfig.WaitingLaneOccupancyGraceSeconds:0.0})",
            $"  --service-minutes-per-vehicle Estimated service minutes for one queued vehicle (default: {AppConfig.DefaultEstimatedServiceMinutesPerVehicle:0.0})",
            $"  --stale-track-seconds Drop tracks after this many seconds without a match (default: {AppConfig.DefaultStaleTrackSeconds:0.0})",
            $"  --stationary-confirm-seconds Seconds a vehicle must stay nearly still in waiting lane (default: {AppConfig.DefaultStationaryConfirmSeconds:0.0})",
            $"  --waiting-stationary-max-displacement-px Max movement to still count as stationary (default: {AppConfig.DefaultWaitingStationaryMaxDisplacementPx:0.0})",
            "  --view          Show OpenCV windows",
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

sealed class RegionPresenceState
{
    public double LastSeenAt { get; set; }
    public int LastCount { get; set; }
    public int LastStationaryCount { get; set; }
    public IReadOnlySet<string> LastStationaryTrackIds { get; set; } = AppConfig.EmptyTrackIdSet;
    public IReadOnlySet<string> LastTrackIds { get; set; } = AppConfig.EmptyTrackIdSet;
    public int LastCarCount { get; set; }
    public int LastBusCount { get; set; }
    public int LastHeavyCount { get; set; }
}

sealed class CameraAnalysis
{
    public required List<TrackedVehicle> RegionVehicles { get; init; }
    public required List<TrackedVehicle> WaitingLaneVehicles { get; init; }
    public required List<TrackedVehicle> StationaryWaitingLaneVehicles { get; init; }
    public required Dictionary<string, IReadOnlySet<string>> RegionTrackIds { get; init; }
    public required Dictionary<string, IReadOnlySet<string>> RegionStationaryTrackIds { get; init; }
    public required Dictionary<string, int> RegionVehicleCounts { get; init; }
    public required Dictionary<string, int> RegionCarCounts { get; init; }
    public required Dictionary<string, int> RegionBusCounts { get; init; }
    public required Dictionary<string, int> RegionHeavyCounts { get; init; }
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
        var totalSeconds = Math.Max(0, (int)seconds);
        var minutes = totalSeconds / 60;
        var remainingSeconds = totalSeconds % 60;
        return $"{minutes:00}:{remainingSeconds:00}";
    }
}




