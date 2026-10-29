# Separate periodic structure and ordered object matching

## Confirmed requirements
- Label 1 contains periodic structure used for alignment and periodic transfer.
- Label 2 contains three different objects A, B, C. Horizontal shifts by a
  label-1 period must not copy one label-2 object into another object's position.
- Target groups can appear as ABC, BCA or CAB. Objects at image edges may be
  only partially visible.
- User confirmed source orders, left to right:
  - 18851_AFE568N05MA_TDAOI520_75_-332.651_-750.175_O_S_20260921_084833.jpg: ABC
  - 18851_AFE568N05UA_TDAOI520_340_-192.681_-455.144_O_S_20260921_082643.jpg: CAB
- Both files currently encode the three objects as label 2 without group IDs.

## Recommended approach
Object-level appearance matching plus order constraints. Keep label-1 periodic
alignment as geometry initialization; identify A/B/C independently using the
source image patches and their individual masks. Validate the joint order against
ABC/BCA/CAB and reject duplicate/conflicting assignments.

Alternative: enumerate group-phase transforms for the whole label-2 union.
This is simpler but forces all objects to share one correction and is less
robust when boundary objects are heavily clipped. Do not use blind single-pitch
horizontal replication for label 2 in either approach.

## Interfaces and data
- Store each template's ordered label-2 identities in per-template configuration,
  sorting label-2 polygons by horizontal position before assigning names.
- Preserve original polygon instance masks, source valid regions, and pixel
  coordinates; do not try to recover object identity from the merged class mask.
- Output semantic class IDs remain 0/1/2. Store A/B/C identity and each match's
  source polygon, matrix, visible fraction and confidence separately.
- Use both templates as reference examples for each identity when available.

## Pipeline
1. Prepare valid image regions before scaling as currently implemented.
2. Register periodic structure, retaining plausible geometry candidates.
3. Transfer/complete label 1 only; periodic copies cannot write label 2.
4. Search reference objects near geometry-compatible target locations using image
   evidence; compare identity alternatives and resolve a coherent ordered group.
5. For an edge object, compare only visible valid pixels, require adequate image
   evidence and support from complete neighboring objects, and clip the warped
   mask to the actual input ROI. Never pad missing pixels as observed background.
6. Merge accepted semantic masks. Absence of label 2 in a label-1 periodic copy
   is not evidence of true background. Keep unresolved coverage unknown rather
   than publishing an artificial 100% valid mask.
7. Persist separate label-1/label-2 matching diagnostics and A/B/C assignments for
   the existing independent result viewer.

## Validation
- Deterministic synthetic ABC, BCA and CAB targets with distinguishable object
  appearances; assert identity, location and semantic mask, not only coverage.
- Clip each identity at both left/right boundaries; vary visible fraction and
  check rejection with insufficient evidence.
- Ensure a label-1 lattice copy cannot produce a false label-2 object or mark
  unresolved label-2 areas as known background.
- Preserve explicit image/annotation validity, uint16 labels and rejected-match
  behavior; run existing tests.
- Inspect actual 1379453 and 1383316_defect predictions separately. No target
  ground truth is available for full identity accuracy, so do not call visual
  inspection or complete ROI coverage an accuracy measurement.

Status: approved by user ("实现它"); implemented and verified. Source identity
order is stored in the SD-1 OpenCV configuration; semantic output remains class 2.
