// Ghost Radar v2.1.0 - On-device YOLO Dataset Exporter
//
// Save current camera frame + bounding boxes as a YOLO-format training sample.
//
// Inspired by CVAT (Intel) and VIA (Oxford VGG) - both let users collect
// bounding-box datasets for object detection training. This brings that
// workflow on-device so the user can build a custom dataset of "interesting"
// detections during clinical review.
//
// File structure saved to app docs/dataset/:
//   classes.txt        // 80 COCO class names, one per line
//   images/img_0001.jpg   // frame snapshot
//   images/img_0002.jpg
//   labels/img_0001.txt   // YOLO format: "class_id cx cy w h" per line (normalized 0..1)
//   labels/img_0002.txt
//   manifest.json      // {count, created_at, coco_classes: [...]}
//
// YOLO bbox format: <class_id> <x_center> <y_center> <width> <height>
// All values normalized to [0, 1] relative to image dimensions.
//
// Export workflow: zip the dataset folder and share via share_plus.

import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

/// Bounding box in image coordinates (pixels).
class ImageBBox {
  const ImageBBox({
    required this.className,
    required this.cx,
    required this.cy,
    required this.w,
    required this.h,
  });
  final String className;
  final double cx; // center x
  final double cy; // center y
  final double w; // width
  final double h; // height
}

class DatasetExporter {
  /// COCO 80 classes (order = class_id 0..79). YOLOv8 default.
  static const List<String> cocoClasses = <String>[
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
  ];

  /// Save a frame + bboxes to the dataset folder.
  /// Returns the basename (e.g., "img_0001") so caller can reference it.
  static Future<String> saveSample({
    required Uint8List jpegBytes,
    required int imageWidth,
    required int imageHeight,
    required List<ImageBBox> bboxes,
  }) async {
    final dir = await _ensureDatasetDir();
    final imagesDir = Directory('${dir.path}/images');
    final labelsDir = Directory('${dir.path}/labels');
    if (!await imagesDir.exists()) await imagesDir.create(recursive: true);
    if (!await labelsDir.exists()) await labelsDir.create(recursive: true);

    // Determine next index by counting existing files
    int nextIdx = 1;
    final existing = await imagesDir.list().toList();
    nextIdx = existing.length + 1;
    final base = 'img_${nextIdx.toString().padLeft(4, '0')}';

    // Write JPEG
    final imgFile = File('${imagesDir.path}/$base.jpg');
    await imgFile.writeAsBytes(jpegBytes, flush: true);

    // Write YOLO label file
    final labelLines = <String>[];
    for (final bbox in bboxes) {
      final int classId = cocoClasses.indexOf(bbox.className);
      if (classId < 0) continue; // unknown class - skip
      // Normalize to 0..1
      final double cx = bbox.cx / imageWidth;
      final double cy = bbox.cy / imageHeight;
      final double w = bbox.w / imageWidth;
      final double h = bbox.h / imageHeight;
      labelLines.add(
        '$classId '
            '${cx.toStringAsFixed(6)} '
            '${cy.toStringAsFixed(6)} '
            '${w.toStringAsFixed(6)} '
            '${h.toStringAsFixed(6)}',
      );
    }
    final labelFile = File('${labelsDir.path}/$base.txt');
    await labelFile.writeAsString(labelLines.join('\n'), flush: true);

    // Update manifest
    await _updateManifest(dir, bboxes.length);

    return base;
  }

  /// Get current sample count.
  static Future<int> getSampleCount() async {
    final dir = await _ensureDatasetDir();
    final imagesDir = Directory('${dir.path}/images');
    if (!await imagesDir.exists()) return 0;
    final files = await imagesDir.list().toList();
    return files.length;
  }

  /// Get total size in bytes.
  static Future<int> getTotalSize() async {
    final dir = await _ensureDatasetDir();
    int total = 0;
    await for (final entity in dir.list(recursive: true)) {
      if (entity is File) {
        total += await entity.length();
      }
    }
    return total;
  }

  /// Clear all saved samples.
  static Future<void> clearAll() async {
    final dir = await _ensureDatasetDir();
    if (await dir.exists()) {
      await for (final entity in dir.list()) {
        if (entity is Directory) {
          await entity.delete(recursive: true);
        } else {
          await entity.delete();
        }
      }
    }
  }

  /// Zip the dataset folder and share it via Android share sheet.
  /// Returns the size of the zip in bytes, or null if no samples.
  static Future<int?> exportAndShare() async {
    final count = await getSampleCount();
    if (count == 0) return null;

    final dir = await _ensureDatasetDir();
    final archive = Archive();

    // Walk all files in dataset dir and add to archive
    await for (final entity in dir.list(recursive: true)) {
      if (entity is File) {
        final relPath = entity.path.substring(dir.path.length + 1).replaceAll('\\', '/');
        final bytes = await entity.readAsBytes();
        archive.addFile(ArchiveFile(relPath, bytes.length, bytes));
      }
    }

    final encoder = ZipEncoder();
    final zipBytes = encoder.encode(archive);

    final tempDir = await getTemporaryDirectory();
    final zipPath =
        '${tempDir.path}/ghost_radar_dataset_${DateTime.now().millisecondsSinceEpoch}.zip';
    final zipFile = File(zipPath);
    await zipFile.writeAsBytes(zipBytes, flush: true);
    final zipSize = zipBytes.length;

    await Share.shareXFiles(
      [XFile(zipPath, mimeType: 'application/zip')],
      text: 'Ghost Radar dataset: $count samples (YOLO format)',
    );
    return zipSize;
  }

  // ====== Internal helpers ======

  static Future<Directory> _ensureDatasetDir() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/dataset');
    if (!await dir.exists()) await dir.create(recursive: true);
    return dir;
  }

  static Future<void> _updateManifest(Directory dir, int newBoxes) async {
    final manifestFile = File('${dir.path}/manifest.json');
    Map<String, dynamic> manifest;
    if (await manifestFile.exists()) {
      final content = await manifestFile.readAsString();
      try {
        manifest = jsonDecode(content) as Map<String, dynamic>;
      } catch (_) {
        manifest = <String, dynamic>{};
      }
    } else {
      manifest = <String, dynamic>{};
    }
    manifest['count'] = (manifest['count'] as int? ?? 0) + 1;
    manifest['total_boxes'] = (manifest['total_boxes'] as int? ?? 0) + newBoxes;
    manifest['updated_at'] = DateTime.now().toIso8601String();
    if (!manifest.containsKey('created_at')) {
      manifest['created_at'] = DateTime.now().toIso8601String();
    }
    if (!manifest.containsKey('coco_classes')) {
      manifest['coco_classes'] = cocoClasses;
    }
    await manifestFile.writeAsString(
      const JsonEncoder.withIndent('  ').convert(manifest),
      flush: true,
    );
  }

  /// Write classes.txt (YOLO convention - class names, one per line).
  static Future<void> writeClassesTxt() async {
    final dir = await _ensureDatasetDir();
    final f = File('${dir.path}/classes.txt');
    if (!await f.exists()) {
      await f.writeAsString(cocoClasses.join('\n'), flush: true);
    }
  }
}
