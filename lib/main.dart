// Ghost Radar - Phân tích tín hiệu âm thanh hạ tần 0-20 Hz + camera âm bản
//
// Audio pipeline:
//   PCM 44.1kHz → 4-stage IIR anti-alias (25Hz)
//                → decimate 441× → 100Hz
//                → 256-point Hann-windowed FFT
//                → 5 band energy (0-1, 1-3, 3-7, 7-15, 15-20Hz)
//                → per-band EMA baseline → alarm nếu band > sensitivity × baseline
//
// Compass (heading):
//   accelerometer (low-pass → gravity) + magnetometer → tilt-compensated heading
//   Mỗi alarm → tạo _Blip tại heading hiện tại
//   Blip fading 8s, blip sáng nhất = "mục tiêu"
//
// Camera:
//   Live preview + ColorFilter.matrix âm bản (invert RGB) + scanlines retro
//   Sync border/glow với alarm
//
// Lưu ý: 1 mic + 1 RGB cam + la bàn KHÔNG cho DOA/IR thật.
// Samsung A17 không có camera IR; "âm bản" chỉ là hiệu ứng giả lập.

import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:record/record.dart';
import 'package:sensors_plus/sensors_plus.dart';

void main() {
  runApp(const GhostRadarApp());
}

class GhostRadarApp extends StatelessWidget {
  const GhostRadarApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Ghost Radar',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF050505),
        primaryColor: const Color(0xFF1FE033),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF1FE033),
          secondary: Color(0xFFFF3B3B),
        ),
        useMaterial3: true,
      ),
      home: const RadarScreen(),
    );
  }
}

class _Blip {
  _Blip({
    required this.angleDeg,
    required this.energy,
    required this.bandIndex,
    required this.detectedAt,
  });
  final double angleDeg; // 0..360, 0=Bắc, 90=Đông
  final double energy; // 0..1
  final int bandIndex;
  final DateTime detectedAt;

  double get ageSeconds =>
      DateTime.now().difference(detectedAt).inMilliseconds / 1000.0;
  double get fade => (1.0 - (ageSeconds / 8.0)).clamp(0.0, 1.0);
  double get score => energy * fade;
}

class RadarScreen extends StatefulWidget {
  const RadarScreen({super.key});

  @override
  State<RadarScreen> createState() => _RadarScreenState();
}

class _RadarScreenState extends State<RadarScreen>
    with SingleTickerProviderStateMixin, WidgetsBindingObserver {
  // ====== Audio: anti-alias + decimation ======
  static const double _lpA = 157.0796 / (44100 + 157.0796);
  double _lp1 = 0, _lp2 = 0, _lp3 = 0, _lp4 = 0;
  int _decCount = 0;
  static const int _decFactor = 441;

  // ====== Low-rate circular buffer (256 @ 100Hz = 2.56s) ======
  static const int _fftSize = 256;
  static const double _binHz = 100 / _fftSize;
  final Float64List _lrBuffer = Float64List(_fftSize);
  int _lrFilled = 0;
  int _lrWriteIdx = 0;
  bool _bufferReady = false;

  late final Float64List _fftRe;
  late final Float64List _fftIm;
  late final Float64List _hann;
  late final Float64List _windowed;

  // ====== 5 bands ======
  static const List<String> _bandNames = <String>[
    '0–1 Hz',
    '1–3 Hz',
    '3–7 Hz',
    '7–15 Hz',
    '15–20 Hz',
  ];
  static const List<List<int>> _bandBins = <List<int>>[
    <int>[0, 1, 2],
    <int>[3, 4, 5, 6, 7],
    <int>[8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    <int>[
      18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37
    ],
    <int>[38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
  ];
  final List<double> _bandEnergy = List<double>.filled(5, 0);
  final List<double> _bandBaseline = List<double>.filled(5, 1.0);
  static const double _emaAlpha = 0.92;
  int _baselineWarmupLeft = 30;

  // ====== Compass (manual tilt-compensated heading) ======
  double _heading = 0;
  bool _hasCompass = false;
  // Smoothed gravity từ accelerometer
  double _gx = 0, _gy = 0, _gz = 0;
  // Magnetometer thô
  double _mx = 0, _my = 0, _mz = 0;
  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<MagnetometerEvent>? _magSub;

  // ====== Camera ======
  CameraController? _cameraController;
  List<CameraDescription> _cameras = <CameraDescription>[];
  String _cameraError = '';

  // ====== Blips ======
  final List<_Blip> _blips = <_Blip>[];
  static const int _maxBlips = 24;
  static const double _blipLifetimeSec = 8.0;
  _Blip? _target;

  // ====== Detection / Alarm ======
  bool _isAlarming = false;
  int _alarmCount = 0;
  String _lastAlarmSummary = '';
  bool _alarmMuted = false;
  Timer? _alarmHapticTimer; // lặp haptic mỗi 350ms trong khi alarm

  // ====== UI state ======
  bool _isScanning = false;
  String _statusText = 'CHƯA KHỞI ĐỘNG';
  String _noteText =
      'Bấm BẮT ĐẦU QUÉT. Cầm điện thoại thẳng đứng, xoay người chậm 360° để dò hướng có năng lượng mạnh nhất. Camera bên cạnh hiển thị ảnh âm bản để soi vùng "nghi ngờ".';
  double _level = 0.0;
  double _angle = 0.0;
  double _sensitivity = 2.5;
  String _dominantBandName = '--';
  double _dominantBandHz = 0;

  // ====== Audio plumbing ======
  AudioRecorder? _recorder;
  StreamSubscription<Uint8List>? _streamSub;
  Timer? _analysisTimer;
  late final Ticker _ticker;

  // ====== Compass direction labels (Vietnamese) ======
  static const List<String> _compassDirs = <String>[
    'B', 'BĐ', 'Đ', 'ĐN', 'N', 'NT', 'T', 'TB',
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _ticker = Ticker(_onTick)..start();

    _hann = Float64List(_fftSize);
    for (int i = 0; i < _fftSize; i++) {
      _hann[i] = 0.5 * (1 - math.cos(2 * math.pi * i / (_fftSize - 1)));
    }
    _fftRe = Float64List(_fftSize);
    _fftIm = Float64List(_fftSize);
    _windowed = Float64List(_fftSize);

    _startSensors();
    _initCamera();
  }

  void _onTick(Duration elapsed) {
    if (!mounted) return;
    setState(() {
      if (_isScanning) {
        _angle = (_angle + 2.5) % 360.0;
      }
    });
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _ticker.dispose();
    _analysisTimer?.cancel();
    _alarmHapticTimer?.cancel();
    _streamSub?.cancel();
    _accelSub?.cancel();
    _magSub?.cancel();
    _recorder?.dispose();
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final c = _cameraController;
    if (c == null || !c.value.isInitialized) return;
    if (state == AppLifecycleState.inactive) {
      c.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  // ====== Sensors (heading manual) ======
  void _startSensors() {
    try {
      _accelSub = accelerometerEventStream(
        samplingPeriod: const Duration(milliseconds: 100),
      ).listen((AccelerometerEvent e) {
        // Low-pass lấy gravity (α nhỏ để mượt)
        const double a = 0.15;
        _gx = a * e.x + (1 - a) * _gx;
        _gy = a * e.y + (1 - a) * _gy;
        _gz = a * e.z + (1 - a) * _gz;
      }, onError: (Object _) {});
    } catch (_) {}

    try {
      _magSub = magnetometerEventStream(
        samplingPeriod: const Duration(milliseconds: 100),
      ).listen((MagnetometerEvent e) {
        _mx = e.x;
        _my = e.y;
        _mz = e.z;
        _updateHeading();
      }, onError: (Object _) {});
    } catch (_) {}
  }

  void _updateHeading() {
    final double gNorm = math.sqrt(_gx * _gx + _gy * _gy + _gz * _gz);
    if (gNorm < 0.1) return;
    final double ngx = _gx / gNorm;
    final double ngy = _gy / gNorm;
    final double ngz = _gz / gNorm;

    final double mNorm = math.sqrt(_mx * _mx + _my * _my + _mz * _mz);
    if (mNorm < 0.1) return;
    final double nmx = _mx / mNorm;
    final double nmy = _my / mNorm;
    final double nmz = _mz / mNorm;

    // East = magnetic × gravity
    final double ex = nmy * ngz - nmz * ngy;
    final double ey = nmz * ngx - nmx * ngz;
    final double ez = nmx * ngy - nmy * ngx;

    // North = gravity × east
    final double ny = ngz * ex - ngx * ez;
    // (nx không dùng trực tiếp, chỉ cần ny cho atan2)

    double headingRad = math.atan2(ey, ny);
    double headingDeg = headingRad * 180 / math.pi;
    if (headingDeg < 0) headingDeg += 360;

    if (headingDeg.isFinite && mounted) {
      setState(() {
        _heading = headingDeg;
        _hasCompass = true;
      });
    }
  }

  // ====== Camera ======
  Future<void> _initCamera() async {
    try {
      final camStatus = await Permission.camera.request();
      if (!camStatus.isGranted) {
        if (mounted) setState(() => _cameraError = 'Không có quyền camera');
        return;
      }
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        if (mounted) setState(() => _cameraError = 'Không tìm thấy camera');
        return;
      }
      // Ưu tiên back camera
      CameraDescription cam = _cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => _cameras.first,
      );
      final controller = CameraController(
        cam,
        ResolutionPreset.low, // 240p, tiết kiệm CPU
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) {
        await controller.dispose();
        return;
      }
      setState(() {
        _cameraController = controller;
        _cameraError = '';
      });
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Lỗi camera: $e');
    }
  }

  // ====== Compass helpers ======
  String _angleToCompass(double deg) {
    final double d = deg % 360;
    final int idx = ((d + 22.5) ~/ 45) % 8;
    return '${d.toStringAsFixed(0)}° ${_compassDirs[idx]}';
  }

  // ====== Permissions ======
  Future<bool> _ensureMicPermission() async {
    final p = await Permission.microphone.request();
    return p.isGranted;
  }

  // ====== Start / Stop scan ======
  Future<void> startScan() async {
    if (_isScanning) return;
    final ok = await _ensureMicPermission();
    if (!ok) {
      setState(() {
        _statusText = 'KHÔNG CÓ QUYỀN MICRO';
        _noteText =
            'Vào Cài đặt → Ứng dụng → Ghost Radar → Quyền → cấp Microphone.';
      });
      return;
    }

    _recorder ??= AudioRecorder();
    try {
      final stream = await _recorder!.startStream(
        const RecordConfig(
          encoder: AudioEncoder.pcm16bits,
          sampleRate: 44100,
          numChannels: 1,
          autoGain: false,
          echoCancel: false,
          noiseSuppress: false,
        ),
      );
      _streamSub = stream.listen(
        _onAudioData,
        onError: (Object e) {
          if (!mounted) return;
          setState(() {
            _statusText = 'LỖI AUDIO STREAM';
            _noteText = '$e';
          });
        },
        cancelOnError: false,
      );
    } catch (e) {
      setState(() {
        _statusText = 'KHÔNG MỞ ĐƯỢC MICRO';
        _noteText = 'Lỗi: $e';
      });
      return;
    }

    _lp1 = _lp2 = _lp3 = _lp4 = 0;
    _decCount = 0;
    _lrFilled = 0;
    _lrWriteIdx = 0;
    _bufferReady = false;
    for (int i = 0; i < 5; i++) {
      _bandEnergy[i] = 0;
      _bandBaseline[i] = 1.0;
    }
    _baselineWarmupLeft = 30;
    _isAlarming = false;
    _alarmCount = 0;
    _lastAlarmSummary = '';
    _alarmMuted = false;
    _blips.clear();
    _target = null;
    _dominantBandName = '--';
    _dominantBandHz = 0;

    _analysisTimer?.cancel();
    _analysisTimer = Timer.periodic(
      const Duration(milliseconds: 200),
      (_) => _analyzeBuffer(),
    );

    setState(() {
      _isScanning = true;
      _statusText = 'ĐANG QUÉT 0–20 Hz';
      _noteText = _hasCompass
          ? 'Đang quét + camera âm bản. Cầm thẳng đứng, xoay người chậm để dò hướng. '
              'Sau ~6s có baseline.'
          : 'Đang quét + camera âm bản. Không đọc được la bàn — hướng blip sẽ không chính xác.';
    });
  }

  Future<void> stopScan() async {
    if (!_isScanning) return;
    _analysisTimer?.cancel();
    _analysisTimer = null;
    _alarmHapticTimer?.cancel();
    _alarmHapticTimer = null;
    try {
      await _streamSub?.cancel();
    } catch (_) {}
    _streamSub = null;
    try {
      await _recorder?.stop();
    } catch (_) {}
    if (!mounted) return;
    setState(() {
      _isScanning = false;
      _isAlarming = false;
      _statusText = 'ĐÃ DỪNG';
      _noteText =
          'Đã dừng. Báo động: $_alarmCount lần · Blip trên radar: ${_blips.length}';
    });
  }

  void toggleMute() => setState(() => _alarmMuted = !_alarmMuted);

  // ====== Audio callback: PCM → anti-alias → decimate → circular buffer ======
  void _onAudioData(Uint8List bytes) {
    if (bytes.isEmpty) return;
    final List<int> ints = _recorder!.convertBytesToInt16(bytes);
    if (ints.isEmpty) return;
    for (int i = 0; i < ints.length; i++) {
      final double x = ints[i] / 32768.0;
      final double y1 = _lpA * x + (1 - _lpA) * _lp1;
      _lp1 = y1;
      final double y2 = _lpA * y1 + (1 - _lpA) * _lp2;
      _lp2 = y2;
      final double y3 = _lpA * y2 + (1 - _lpA) * _lp3;
      _lp3 = y3;
      final double y4 = _lpA * y3 + (1 - _lpA) * _lp4;
      _lp4 = y4;

      _decCount++;
      if (_decCount >= _decFactor) {
        _decCount = 0;
        _lrBuffer[_lrWriteIdx] = y4;
        _lrWriteIdx = (_lrWriteIdx + 1) % _fftSize;
        if (_lrFilled < _fftSize) {
          _lrFilled++;
          if (_lrFilled == _fftSize) _bufferReady = true;
        }
      }
    }
  }

  void _linearize(Float64List dst) {
    int idx = _lrWriteIdx;
    for (int i = 0; i < _fftSize; i++) {
      dst[i] = _lrBuffer[idx];
      idx = (idx + 1) % _fftSize;
    }
  }

  void _fft() {
    final n = _fftSize;
    int j = 0;
    for (int i = 0; i < n; i++) {
      if (i < j) {
        final double tr = _fftRe[i];
        _fftRe[i] = _fftRe[j];
        _fftRe[j] = tr;
        final double ti = _fftIm[i];
        _fftIm[i] = _fftIm[j];
        _fftIm[j] = ti;
      }
      int m = n >> 1;
      while (m >= 1 && j >= m) {
        j -= m;
        m >>= 1;
      }
      j += m;
    }
    for (int size = 2; size <= n; size <<= 1) {
      final int half = size >> 1;
      final double tablestep = -2 * math.pi / size;
      for (int i = 0; i < n; i += size) {
        for (int k = 0; k < half; k++) {
          final double angle = k * tablestep;
          final double cos = math.cos(angle);
          final double sin = math.sin(angle);
          final int a = i + k;
          final int b = a + half;
          final double tre = _fftRe[b] * cos - _fftIm[b] * sin;
          final double tim = _fftRe[b] * sin + _fftIm[b] * cos;
          _fftRe[b] = _fftRe[a] - tre;
          _fftIm[b] = _fftIm[a] - tim;
          _fftRe[a] += tre;
          _fftIm[a] += tim;
        }
      }
    }
  }

  // ====== Analysis ======
  void _analyzeBuffer() {
    if (!_isScanning || !_bufferReady) return;

    _linearize(_windowed);

    double mean = 0;
    for (int i = 0; i < _fftSize; i++) {
      mean += _windowed[i];
    }
    mean /= _fftSize;
    for (int i = 0; i < _fftSize; i++) {
      _fftRe[i] = (_windowed[i] - mean) * _hann[i];
      _fftIm[i] = 0;
    }

    _fft();

    final int nyquistBin = _fftSize >> 1;
    final List<double> mag2 = List<double>.filled(5, 0);
    int peakBin = -1;
    double peakVal = 0;
    for (int b = 1; b < nyquistBin; b++) {
      final double re = _fftRe[b];
      final double im = _fftIm[b];
      final double m2 = re * re + im * im;
      for (int band = 0; band < 5; band++) {
        if (_bandBins[band].contains(b)) {
          mag2[band] += m2;
          break;
        }
      }
      if (b <= 51 && m2 > peakVal) {
        peakVal = m2;
        peakBin = b;
      }
    }

    for (int i = 0; i < 5; i++) {
      _bandEnergy[i] = mag2[i] / _fftSize;
    }

    for (int i = 0; i < 5; i++) {
      if (_baselineWarmupLeft > 0) {
        _bandBaseline[i] = math.max(_bandEnergy[i], 1e-12);
      } else {
        _bandBaseline[i] =
            _emaAlpha * _bandBaseline[i] + (1 - _emaAlpha) * _bandEnergy[i];
        if (_bandBaseline[i] < 1e-12) _bandBaseline[i] = 1e-12;
      }
    }
    if (_baselineWarmupLeft > 0) _baselineWarmupLeft--;

    int bestBand = -1;
    double bestRatio = 0;
    double bestHz = 0;
    for (int i = 0; i < 5; i++) {
      final double r = _bandEnergy[i] / _bandBaseline[i];
      if (r > bestRatio) {
        bestRatio = r;
        bestBand = i;
        if (peakBin >= _bandBins[i].first && peakBin <= _bandBins[i].last) {
          bestHz = peakBin * _binHz;
        } else {
          bestHz = ((_bandBins[i].first + _bandBins[i].last) / 2.0) * _binHz;
        }
      }
    }

    final double level01 =
        ((bestRatio - 1.0) / math.max(0.5, _sensitivity - 1.0)).clamp(0.0, 1.0);

    String status;
    String note;
    bool newAlarm = false;
    if (_baselineWarmupLeft > 0) {
      status = '⏳ ĐANG THU BASELINE';
      note =
          'Còn ${(_baselineWarmupLeft * 0.2).toStringAsFixed(0)}s để thiết lập đường nền. Hãy giữ yên tĩnh.';
    } else if (bestRatio >= _sensitivity) {
      status = '🚨 BÁO ĐỘNG ${_bandNames[bestBand]}';
      note =
          'Phát hiện năng lượng ${bestRatio.toStringAsFixed(1)}× đường nền '
          'trong dải ${_bandNames[bestBand]} (≈ ${bestHz.toStringAsFixed(1)} Hz).';
      newAlarm = true;
    } else if (bestRatio >= _sensitivity * 0.6) {
      status = '⚠ CÓ TÍN HIỆU';
      note =
          'Tín hiệu nhô lên ở ${_bandNames[bestBand]} '
          '(${(bestRatio).toStringAsFixed(1)}× nền, ≈ ${bestHz.toStringAsFixed(1)} Hz).';
    } else {
      status = '✓ NỀN THẤP';
      note =
          'Yên tĩnh. Đang quét 0–20 Hz, ngưỡng = ${_sensitivity.toStringAsFixed(1)}× nền.';
    }

    _blips.removeWhere((b) => b.ageSeconds > _blipLifetimeSec);
    _Blip? best;
    for (final b in _blips) {
      if (best == null || b.score > best.score) best = b;
    }
    _target = best;

    if (!mounted) return;
    setState(() {
      _level = level01;
      _statusText = status;
      _noteText = note;
      _dominantBandName = bestBand >= 0 ? _bandNames[bestBand] : '--';
      _dominantBandHz = bestHz;
    });

    if (newAlarm) {
      _triggerAlarm(bestBand, bestHz, bestRatio);
    } else if (_isAlarming) {
      _stopAlarm();
    }
  }

  void _triggerAlarm(int band, double hz, double ratio) {
    if (!_isAlarming) {
      _isAlarming = true;
      _alarmCount += 1;
      _lastAlarmSummary =
          '${_bandNames[band]} · ≈ ${hz.toStringAsFixed(1)} Hz · ${ratio.toStringAsFixed(1)}× nền';
      // Bắt đầu vòng lặp haptic: check mute state mỗi tick
      _alarmHapticTimer?.cancel();
      _alarmHapticTimer = Timer.periodic(
        const Duration(milliseconds: 350),
        (_) {
          if (!_isAlarming || !mounted) return;
          if (!_alarmMuted) {
            HapticFeedback.heavyImpact();
            SystemSound.play(SystemSoundType.click);
          }
        },
      );
    } else {
      _lastAlarmSummary =
          '${_bandNames[band]} · ≈ ${hz.toStringAsFixed(1)} Hz · ${ratio.toStringAsFixed(1)}× nền';
    }

    final double energy01 =
        (ratio / math.max(2.0, _sensitivity)).clamp(0.0, 1.0);
    _blips.add(_Blip(
      angleDeg: _heading,
      energy: energy01,
      bandIndex: band,
      detectedAt: DateTime.now(),
    ));
    if (_blips.length > _maxBlips) {
      _blips.removeRange(0, _blips.length - _maxBlips);
    }
  }

  void _stopAlarm() {
    _isAlarming = false;
    _alarmHapticTimer?.cancel();
    _alarmHapticTimer = null;
  }

  // ====== UI ======
  @override
  Widget build(BuildContext context) {
    final bool alarm = _isAlarming;
    return Scaffold(
      appBar: AppBar(
        title: const Text('GHOST RADAR · 0–20 Hz · IR'),
        centerTitle: true,
        backgroundColor: Colors.black,
        elevation: 0,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(10.0),
          child: Column(
            children: [
              _alarmBanner(),
              // ===== 2 cột: radar (trái) + camera âm bản (phải) =====
              SizedBox(
                height: 270,
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(child: _radarPanel(alarm)),
                    const SizedBox(width: 8),
                    Expanded(child: _cameraPanel(alarm)),
                  ],
                ),
              ),
              const SizedBox(height: 8),
              _infoPanel(),
              const SizedBox(height: 8),
              _bandChart(),
              const SizedBox(height: 8),
              _sensitivityRow(),
              const SizedBox(height: 6),
              _navigationPanel(),
              const SizedBox(height: 6),
              _note(),
              const SizedBox(height: 6),
              _buttons(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _alarmBanner() {
    if (!_isAlarming) return const SizedBox.shrink();
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 10),
      decoration: BoxDecoration(
        color: const Color(0xFFFF3B3B),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.warning_amber_rounded, color: Colors.white, size: 24),
          const SizedBox(width: 6),
          Expanded(
            child: Text(
              '🚨 $_lastAlarmSummary',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _radarPanel(bool alarm) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 120),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: alarm
              ? const Color(0xFFFF3B3B)
              : const Color(0xFF1FE033).withOpacity(0.5),
          width: alarm ? 4.0 : 1.5,
        ),
        boxShadow: alarm
            ? [
                BoxShadow(
                  color: const Color(0xFFFF3B3B).withOpacity(0.45),
                  blurRadius: 20,
                  spreadRadius: 2,
                ),
              ]
            : null,
      ),
      child: Stack(
        children: [
          Positioned.fill(
            child: CustomPaint(
              painter: _RadarPainter(
                sweepAngleDeg: _angle,
                headingDeg: _heading,
                level: _level,
                alarm: alarm,
                blips: List<_Blip>.unmodifiable(_blips),
                target: _target,
                hasCompass: _hasCompass,
              ),
            ),
          ),
          Positioned(
            top: 4,
            left: 4,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
              color: const Color(0xFF1FE033).withOpacity(0.75),
              child: const Text('RADAR',
                  style: TextStyle(
                      fontSize: 9,
                      color: Colors.black,
                      fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _cameraPanel(bool alarm) {
    final Color borderColor = alarm
        ? const Color(0xFFFF3B3B)
        : const Color(0xFF1FE033).withOpacity(0.5);
    return AnimatedContainer(
      duration: const Duration(milliseconds: 120),
      decoration: BoxDecoration(
        color: Colors.black,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: borderColor,
          width: alarm ? 4.0 : 1.5,
        ),
        boxShadow: alarm
            ? [
                BoxShadow(
                  color: const Color(0xFFFF3B3B).withOpacity(0.45),
                  blurRadius: 20,
                  spreadRadius: 2,
                ),
              ]
            : null,
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(11),
        child: Stack(
          fit: StackFit.expand,
          children: [
            _cameraContent(),
            // Scanlines retro
            IgnorePointer(
              child: Column(
                children: List<Widget>.generate(60, (i) {
                  return Expanded(
                    child: Container(
                      color: i.isEven
                          ? Colors.transparent
                          : Colors.black.withOpacity(0.30),
                    ),
                  );
                }),
              ),
            ),
            // Label
            Positioned(
              top: 4,
              left: 4,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                color: alarm
                    ? const Color(0xFFFF3B3B).withOpacity(0.85)
                    : const Color(0xFF1FE033).withOpacity(0.75),
                child: Text(
                  alarm ? 'IR · ALARM' : 'IR · ÂM BẢN',
                  style: const TextStyle(
                      fontSize: 9,
                      color: Colors.black,
                      fontWeight: FontWeight.bold),
                ),
              ),
            ),
            // Crosshair overlay
            Center(
              child: IgnorePointer(
                child: SizedBox(
                  width: 50,
                  height: 50,
                  child: CustomPaint(painter: _CrosshairPainter(alarm: alarm)),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _cameraContent() {
    final ctrl = _cameraController;
    if (_cameraError.isNotEmpty) {
      return Container(
        color: const Color(0xFF0A0A0A),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text(
              _cameraError,
              textAlign: TextAlign.center,
              style: const TextStyle(
                  color: Color(0xFFFF8080), fontSize: 11),
            ),
          ),
        ),
      );
    }
    if (ctrl == null || !ctrl.value.isInitialized) {
      return Container(
        color: const Color(0xFF0A0A0A),
        child: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor:
                      AlwaysStoppedAnimation<Color>(Color(0xFF1FE033)),
                ),
              ),
              SizedBox(height: 6),
              Text('Đang mở camera…',
                  style:
                      TextStyle(fontSize: 10, color: Color(0xFF808080))),
            ],
          ),
        ),
      );
    }
    return FittedBox(
      fit: BoxFit.cover,
      child: SizedBox(
        width: ctrl.value.previewSize?.height ?? 240,
        height: ctrl.value.previewSize?.width ?? 240,
        child: ColorFiltered(
          colorFilter: const ColorFilter.matrix(<double>[
            -1, 0, 0, 0, 255, //
            0, -1, 0, 0, 255, //
            0, 0, -1, 0, 255, //
            0, 0, 0, 1, 0, //
          ]),
          child: CameraPreview(ctrl),
        ),
      ),
    );
  }

  Widget _infoPanel() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: const Color(0xFF101010),
        border: Border.all(
          color: _isAlarming
              ? const Color(0xFFFF3B3B)
              : const Color(0xFF1FE033).withOpacity(0.5),
        ),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            _statusText,
            style: TextStyle(
              fontSize: 16,
              color: _isAlarming
                  ? const Color(0xFFFF3B3B)
                  : const Color(0xFF1FE033),
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Dải mạnh nhất: $_dominantBandName · ≈ ${_dominantBandHz.toStringAsFixed(2)} Hz',
            style: const TextStyle(fontSize: 13),
          ),
          Text(
            '🧭 Bạn đang quay mặt về: ${_angleToCompass(_heading)}'
            '${!_hasCompass ? " (không có la bàn)" : ""}',
            style: const TextStyle(fontSize: 13),
          ),
          if (_isScanning)
            Text(
              'Báo động: $_alarmCount lần · Blip trên radar: ${_blips.length}'
              ' · Camera: ${_cameraController?.value.isInitialized == true ? "ON" : "OFF"}',
              style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
            ),
        ],
      ),
    );
  }

  Widget _bandChart() {
    double maxE = 0;
    for (final e in _bandEnergy) {
      if (e > maxE) maxE = e;
    }
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF0E0E0E),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0xFF252525)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Padding(
            padding: EdgeInsets.only(left: 4, bottom: 6),
            child: Text('NĂNG LƯỢNG 5 DẢI TẦN (0–20 Hz)',
                style: TextStyle(
                    fontSize: 12,
                    color: Color(0xFFB0B0B0),
                    fontWeight: FontWeight.bold)),
          ),
          for (int i = 0; i < 5; i++) _bandBar(i, maxE),
        ],
      ),
    );
  }

  Widget _bandBar(int i, double maxE) {
    final double e = _bandEnergy[i];
    final double b = _bandBaseline[i];
    final double ratio = b > 0 ? e / b : 0;
    final double widthFrac = maxE > 0 ? (e / maxE).clamp(0.0, 1.0) : 0.0;
    final Color barColor = ratio >= _sensitivity
        ? const Color(0xFFFF3B3B)
        : ratio >= _sensitivity * 0.6
            ? const Color(0xFFFFB300)
            : const Color(0xFF1FE033);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(
              _bandNames[i],
              style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
            ),
          ),
          Expanded(
            child: Stack(
              children: [
                Container(
                  height: 16,
                  decoration: BoxDecoration(
                    color: const Color(0xFF1A1A1A),
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
                FractionallySizedBox(
                  widthFactor: widthFrac,
                  child: Container(
                    height: 16,
                    decoration: BoxDecoration(
                      color: barColor,
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(width: 6),
          SizedBox(
            width: 56,
            child: Text(
              '${ratio.toStringAsFixed(1)}×',
              textAlign: TextAlign.right,
              style: const TextStyle(fontSize: 11, color: Color(0xFF808080)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sensitivityRow() {
    return Row(
      children: [
        const Text('Độ nhạy', style: TextStyle(fontSize: 13)),
        const SizedBox(width: 8),
        Expanded(
          child: SliderTheme(
            data: SliderTheme.of(context).copyWith(
              activeTrackColor: const Color(0xFF1FE033),
              inactiveTrackColor: const Color(0xFF303030),
              thumbColor: const Color(0xFF1FE033),
            ),
            child: Slider(
              min: 1.2,
              max: 6.0,
              divisions: 24,
              value: _sensitivity,
              label: '${_sensitivity.toStringAsFixed(1)}×',
              onChanged: _isScanning
                  ? (v) => setState(() => _sensitivity = v)
                  : null,
            ),
          ),
        ),
        SizedBox(
          width: 42,
          child: Text(
            '${_sensitivity.toStringAsFixed(1)}×',
            textAlign: TextAlign.right,
            style: const TextStyle(fontSize: 13, color: Color(0xFFB0B0B0)),
          ),
        ),
      ],
    );
  }

  Widget _navigationPanel() {
    if (_target == null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: const Color(0xFF0B0B0B),
          borderRadius: BorderRadius.circular(6),
          border: Border.all(color: const Color(0xFF252525)),
        ),
        child: const Row(
          children: [
            Icon(Icons.explore, color: Color(0xFF606060), size: 18),
            SizedBox(width: 6),
            Expanded(
              child: Text(
                '🎯 Mục tiêu: chưa có. Bấm BẮT ĐẦU, xoay người chậm 360° và chờ blip đỏ đầu tiên.',
                style: TextStyle(fontSize: 12, color: Color(0xFF808080)),
              ),
            ),
          ],
        ),
      );
    }
    final t = _target!;
    double rel = t.angleDeg - _heading;
    while (rel > 180) {
      rel -= 360;
    }
    while (rel < -180) {
      rel += 360;
    }
    final String turn = rel.abs() < 8
        ? 'Đang đối diện'
        : rel > 0
            ? 'Xoay phải ${rel.toStringAsFixed(0)}°'
            : 'Xoay trái ${rel.abs().toStringAsFixed(0)}°';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF1A0808),
        borderRadius: BorderRadius.circular(6),
        border: Border.all(color: const Color(0xFFFF3B3B).withOpacity(0.6)),
      ),
      child: Row(
        children: [
          const Icon(Icons.gps_fixed, color: Color(0xFFFF3B3B), size: 22),
          const SizedBox(width: 6),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  '🎯 Mục tiêu: ${_angleToCompass(t.angleDeg)}  '
                  '(${_bandNames[t.bandIndex]})',
                  style: const TextStyle(
                      fontSize: 13,
                      color: Color(0xFFFF8080),
                      fontWeight: FontWeight.bold),
                ),
                Text(
                  '$turn · ${(t.energy * _sensitivity).toStringAsFixed(1)}× nền'
                  ' · còn ${(_blipLifetimeSec - t.ageSeconds).toStringAsFixed(0)}s',
                  style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _note() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: const Color(0xFF0B0B0B),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        _noteText,
        textAlign: TextAlign.center,
        style: const TextStyle(fontSize: 12, color: Color(0xFFB0B0B0)),
      ),
    );
  }

  Widget _buttons() {
    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isScanning ? null : startScan,
            icon: const Icon(Icons.play_arrow),
            label: const Text('BẮT ĐẦU'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF1FE033),
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isAlarming ? toggleMute : null,
            icon: Icon(
                _alarmMuted ? Icons.volume_off : Icons.notifications_active),
            label: Text(_alarmMuted ? 'BẬT LOA' : 'TẮT LOA'),
            style: ElevatedButton.styleFrom(
              backgroundColor: _alarmMuted
                  ? const Color(0xFF404040)
                  : const Color(0xFFFFB300),
              foregroundColor: Colors.black,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.bold),
            ),
          ),
        ),
        const SizedBox(width: 6),
        Expanded(
          child: ElevatedButton.icon(
            onPressed: _isScanning ? stopScan : null,
            icon: const Icon(Icons.stop),
            label: const Text('DỪNG'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFFF3B3B),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 11),
              textStyle: const TextStyle(
                  fontSize: 14, fontWeight: FontWeight.bold),
            ),
          ),
        ),
      ],
    );
  }
}

// ============================================================
// Radar painter
// ============================================================
class _RadarPainter extends CustomPainter {
  _RadarPainter({
    required this.sweepAngleDeg,
    required this.headingDeg,
    required this.level,
    required this.alarm,
    required this.blips,
    required this.target,
    required this.hasCompass,
  });
  final double sweepAngleDeg;
  final double headingDeg;
  final double level;
  final bool alarm;
  final List<_Blip> blips;
  final _Blip? target;
  final bool hasCompass;

  double _compassToMath(double compassDeg) =>
      (compassDeg - 90) * math.pi / 180.0;

  @override
  void paint(Canvas canvas, Size size) {
    final Offset c = Offset(size.width / 2, size.height / 2);
    final double rMax = math.min(size.width, size.height) * 0.40;

    final ringColor =
        alarm ? const Color(0xFFFF3B3B) : const Color(0xFF1FE033);
    final ringPaint = Paint()
      ..color = ringColor.withOpacity(0.4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.0;
    final crossPaint = Paint()
      ..color = ringColor.withOpacity(0.35)
      ..strokeWidth = 0.7;
    final outerPaint = Paint()
      ..color = ringColor.withOpacity(0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = alarm ? 2.5 : 1.5;

    for (final frac in [0.33, 0.66, 1.0]) {
      canvas.drawCircle(c, rMax * frac, ringPaint);
    }
    canvas.drawLine(
        Offset(c.dx - rMax, c.dy), Offset(c.dx + rMax, c.dy), crossPaint);
    canvas.drawLine(
        Offset(c.dx, c.dy - rMax), Offset(c.dx, c.dy + rMax), crossPaint);
    canvas.drawCircle(c, rMax, outerPaint);

    _drawCompassLabels(canvas, c, rMax);

    // Decorative sweep arc (30°)
    final sweepPaint = Paint()
      ..shader = SweepGradient(
        startAngle: 0,
        endAngle: math.pi / 6,
        colors: <Color>[
          ringColor.withOpacity(0.0),
          ringColor.withOpacity(0.0),
          ringColor.withOpacity(0.35),
        ],
        stops: const <double>[0.0, 0.7, 1.0],
        transform:
            GradientRotation(_compassToMath(sweepAngleDeg) + math.pi / 2),
      ).createShader(Rect.fromCircle(center: c, radius: rMax));
    final sweepPath = Path()
      ..moveTo(c.dx, c.dy)
      ..lineTo(c.dx + rMax, c.dy)
      ..arcTo(
        Rect.fromCircle(center: c, radius: rMax),
        0,
        math.pi / 6,
        false,
      )
      ..close();
    canvas.drawPath(sweepPath, sweepPaint);

    for (final b in blips) {
      _drawBlip(canvas, c, rMax, b, alarm);
    }

    if (target != null) {
      _drawTarget(canvas, c, rMax, target!, alarm);
    }

    _drawHeadingNeedle(canvas, c, rMax, alarm);

    final centerPaint = Paint()
      ..color = (alarm ? const Color(0xFFFF3B3B) : const Color(0xFF1FE033))
          .withOpacity(level > 0.05 || alarm ? 0.95 : 0.6);
    final dotR = 5.0 + level.clamp(0.0, 1.0) * 10.0;
    canvas.drawCircle(c, dotR, centerPaint);
  }

  void _drawCompassLabels(Canvas canvas, Offset c, double r) {
    final labelR = r + 12;
    const style = TextStyle(
      color: Color(0xFF808080),
      fontSize: 11,
      fontWeight: FontWeight.bold,
    );
    const dirs = ['B', 'Đ', 'N', 'T'];
    final angles = <double>[0, 90, 180, 270];
    for (int i = 0; i < 4; i++) {
      final a = _compassToMath(angles[i]);
      final dx = c.dx + labelR * math.cos(a);
      final dy = c.dy + labelR * math.sin(a);
      final tp = TextPainter(
        text: TextSpan(text: dirs[i], style: style),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(canvas, Offset(dx - tp.width / 2, dy - tp.height / 2));
    }
  }

  void _drawBlip(Canvas canvas, Offset c, double rMax, _Blip b, bool alarm) {
    final double a = _compassToMath(b.angleDeg);
    final double rr = rMax * 0.7;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);
    final double size = 6.0 + b.energy * 14.0;
    final double alpha = b.fade * (alarm ? 1.0 : 0.95);

    final Color col = b.score > 0.7
        ? const Color(0xFFFF3B3B)
        : b.score > 0.4
            ? const Color(0xFFFFB300)
            : const Color(0xFF1FE033);

    final glow = Paint()
      ..color = col.withOpacity(alpha * 0.35)
      ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 6);
    canvas.drawCircle(Offset(x, y), size * 1.6, glow);

    final core = Paint()..color = col.withOpacity(alpha);
    canvas.drawCircle(Offset(x, y), size, core);

    final inner = Paint()..color = Colors.white.withOpacity(alpha * 0.9);
    canvas.drawCircle(Offset(x, y), size * 0.35, inner);
  }

  void _drawTarget(
      Canvas canvas, Offset c, double rMax, _Blip t, bool alarm) {
    final double a = _compassToMath(t.angleDeg);
    final double rr = rMax * 0.7;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);

    final ring = Paint()
      ..color = const Color(0xFFFF3B3B).withOpacity(0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
    canvas.drawCircle(Offset(x, y), 22, ring);

    final cross = Paint()
      ..color = const Color(0xFFFF3B3B).withOpacity(0.85)
      ..strokeWidth = 1.5;
    canvas.drawLine(Offset(x - 26, y), Offset(x - 8, y), cross);
    canvas.drawLine(Offset(x + 8, y), Offset(x + 26, y), cross);
    canvas.drawLine(Offset(x, y - 26), Offset(x, y - 8), cross);
    canvas.drawLine(Offset(x, y + 8), Offset(x, y + 26), cross);

    final arrowPaint = Paint()
      ..color = const Color(0xFFFF8080).withOpacity(0.9)
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;
    final double startX = c.dx + (rMax * 0.05) * math.cos(a);
    final double startY = c.dy + (rMax * 0.05) * math.sin(a);
    final double endX = x - 30 * math.cos(a);
    final double endY = y - 30 * math.sin(a);
    final path = Path()
      ..moveTo(startX, startY)
      ..lineTo(endX, endY);
    canvas.drawPath(path, arrowPaint);
    final headLen = 10.0;
    final headAngle = 0.5;
    final ax1 = endX - headLen * math.cos(a - headAngle);
    final ay1 = endY - headLen * math.sin(a - headAngle);
    final ax2 = endX - headLen * math.cos(a + headAngle);
    final ay2 = endY - headLen * math.sin(a + headAngle);
    final head = Path()
      ..moveTo(endX, endY)
      ..lineTo(ax1, ay1)
      ..lineTo(ax2, ay2)
      ..close();
    canvas.drawPath(
        head, Paint()..color = const Color(0xFFFF8080).withOpacity(0.9));
  }

  void _drawHeadingNeedle(Canvas canvas, Offset c, double rMax, bool alarm) {
    final double a = _compassToMath(headingDeg);
    final double rr = rMax * 0.92;
    final double x = c.dx + rr * math.cos(a);
    final double y = c.dy + rr * math.sin(a);
    final paint = Paint()
      ..color = (alarm
              ? const Color(0xFFFF8080)
              : const Color(0xFF80FF80))
          .withOpacity(0.7)
      ..strokeWidth = 2.0;
    canvas.drawLine(c, Offset(x, y), paint);
    final headLen = 12.0;
    final headAngle = 0.45;
    final ax1 = x - headLen * math.cos(a - headAngle);
    final ay1 = y - headLen * math.sin(a - headAngle);
    final ax2 = x - headLen * math.cos(a + headAngle);
    final ay2 = y - headLen * math.sin(a + headAngle);
    final head = Path()
      ..moveTo(x, y)
      ..lineTo(ax1, ay1)
      ..lineTo(ax2, ay2)
      ..close();
    canvas.drawPath(head, Paint()..color = paint.color);
  }

  @override
  bool shouldRepaint(covariant _RadarPainter old) =>
      old.sweepAngleDeg != sweepAngleDeg ||
      old.headingDeg != headingDeg ||
      old.level != level ||
      old.alarm != alarm ||
      old.blips.length != blips.length ||
      old.target != target ||
      old.hasCompass != hasCompass;
}

// ============================================================
// Crosshair overlay cho camera panel
// ============================================================
class _CrosshairPainter extends CustomPainter {
  _CrosshairPainter({required this.alarm});
  final bool alarm;
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = (alarm ? const Color(0xFFFF3B3B) : const Color(0xFF80FF80))
          .withOpacity(0.5)
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;
    final cx = size.width / 2;
    final cy = size.height / 2;
    canvas.drawCircle(Offset(cx, cy), size.width * 0.35, paint);
    canvas.drawLine(Offset(0, cy), Offset(size.width, cy), paint);
    canvas.drawLine(Offset(cx, 0), Offset(cx, size.height), paint);
  }

  @override
  bool shouldRepaint(covariant _CrosshairPainter old) => old.alarm != alarm;
}
