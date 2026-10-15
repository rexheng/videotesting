// Single source of tunable constants. Later modules import from here; nothing is duplicated.

// Pinned versions of the CDN-loaded libraries.
export const VERSIONS = { mediapipe: '1.0.1', ort: '1.30.0' };

// Face recognition (SFace cosine similarity).
export const FACE_MATCH = 0.363;
export const FACE_UNSURE = 0.30;
export const EMBED_RECHECK_MS = 1000;
export const MAX_EMBEDS_PER_FRAME = 2;

// Box tracking between frames.
export const TRACK_IOU = 0.3;
export const TRACK_TTL_MS = 1000;

// Shirt colour matching.
export const SHIRT_MATCH = 0.75;
export const SHIRT_MARGIN = 0.10;
export const SHIRT_BLEND = 0.3;

// Detector limits.
export const MAX_FACES = 4;
export const MAX_POSES = 4;

// Head-turn thresholds, in degrees of yaw.
export const YAW_STRAIGHT = 12;
export const YAW_TURNED = 25;

// Sign-up capture.
export const SAMPLES_PER_STEP = 5;
export const SAMPLE_GAP_MS = 150;
export const NO_FACE_TIP_MS = 3000;
export const STUCK_TIP_MS = 8000;

// Torso region used for shirt sampling.
export const SHOULDER_VISIBILITY = 0.6;
export const HIP_VISIBILITY = 0.5;
export const MIN_TORSO_PX = 40;
export const TORSO_SHRINK = 0.15;
export const MAX_SHIRT_SAMPLES = 2000;

// Persistence and UI.
export const STORAGE_KEY = 'videotesting.people.v1';
export const NAME_MAX = 30;
export const LABEL_COLOURS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'];
