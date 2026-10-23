// Single source of tunable constants. Later modules import from here; nothing is duplicated.

// Pinned versions of the CDN-loaded libraries.
export const VERSIONS = { mediapipe: '1.0.1', ort: '1.30.0' };

// Face recognition (SFace cosine similarity).
export const FACE_MATCH = 0.363;
export const FACE_UNSURE = 0.30;
export const EMBED_RECHECK_MS = 1000;
export const UNKNOWN_RECHECK_MS = 250;
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
export const ENROL_DONE_MS = 2000; // how long the "Done!" message stays before the panel closes

// Torso region used for shirt sampling.
export const SHOULDER_VISIBILITY = 0.6;
export const HIP_VISIBILITY = 0.5;
export const BODY_VISIBILITY = 0.5;
export const MIN_TORSO_PX = 40;
export const TORSO_SHRINK = 0.15;
export const MAX_SHIRT_SAMPLES = 2000;

// Shirt Check: skin versus cloth inside the torso (selfie multiclass categories).
// Shirtless when skin is at least this share of (skin + cloth); still dressed at or below the other.
export const SHIRTCHECK_SHIRTLESS_AT = 0.6;
export const SHIRTCHECK_DRESSED_AT = 0.4;
export const SHIRTCHECK_MIN_SAMPLES = 40;
export const SHIRTCHECK_MIN_FACE = 0.08; // normalised box width and height
export const SHIRTCHECK_FACE_HOLD_MS = 800;
export const SHIRTCHECK_DARE_MS = 25000;
export const SHIRTCHECK_HOLD_MS = 1000;
export const SHIRTCHECK_FACE_GRACE_MS = 700;
export const SHIRTCHECK_SAMPLE_GAP_MS = 150;

// Persistence and UI.
export const STORAGE_KEY = 'videotesting.people.v1';
export const NAME_MAX = 30;
export const LABEL_COLOURS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45'];
