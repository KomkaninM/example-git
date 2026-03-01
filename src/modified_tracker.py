# Lastest Updated by ME21 - Komkanin M. (1 Feb 2026, 16:36)
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils import LOGGER
from ..utils.ops import xywh2ltwh
from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH

# --- ME21 : START ---
import os
import pickle
# --- ME21 : END ---


class STrack(BaseTrack):
    """Single object tracking representation that uses Kalman filtering for state estimation.

    This class is responsible for storing all the information regarding individual tracklets and performs state updates
    and predictions based on Kalman filter.

    Attributes:
        shared_kalman (KalmanFilterXYAH): Shared Kalman filter used across all STrack instances for prediction.
        _tlwh (np.ndarray): Private attribute to store top-left corner coordinates and width and height of bounding box.
        kalman_filter (KalmanFilterXYAH): Instance of Kalman filter used for this particular object track.
        mean (np.ndarray): Mean state estimate vector.
        covariance (np.ndarray): Covariance of state estimate.
        is_activated (bool): Boolean flag indicating if the track has been activated.
        score (float): Confidence score of the track.
        tracklet_len (int): Length of the tracklet.
        cls (Any): Class label for the object.
        idx (int): Index or identifier for the object.
        frame_id (int): Current frame ID.
        start_frame (int): Frame where the object was first detected.
        angle (float | None): Optional angle information for oriented bounding boxes.

    Methods:
        predict: Predict the next state of the object using Kalman filter.
        multi_predict: Predict the next states for multiple tracks.
        multi_gmc: Update multiple track states using a homography matrix.
        activate: Activate a new tracklet.
        re_activate: Reactivate a previously lost tracklet.
        update: Update the state of a matched track.
        convert_coords: Convert bounding box to x-y-aspect-height format.
        tlwh_to_xyah: Convert tlwh bounding box to xyah format.

    Examples:
        Initialize and activate a new track
        >>> track = STrack(xywh=[100, 200, 50, 80, 0], score=0.9, cls="person")
        >>> track.activate(kalman_filter=KalmanFilterXYAH(), frame_id=1)
    """

    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: list[float], score: float, cls: Any):
        """Initialize a new STrack instance.

        Args:
            xywh (list[float]): Bounding box in `(x, y, w, h, idx)` or `(x, y, w, h, angle, idx)` format, where (x, y)
                is the center, (w, h) are width and height, and `idx` is the detection index.
            score (float): Confidence score of the detection.
            cls (Any): Class label for the detected object.
        """
        super().__init__()
        # xywh+idx or xywha+idx
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None 
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        """Predict the next state (mean and covariance) of the object using the Kalman filter."""
        # [ME21] This guesses where the object should be in the next frame.
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0 # [ME21] If an object is lost/untracked, we assume its height isn't changing size rapidly, so we zero out that velocity to stabilize the math.
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks: list[STrack]):
        """Perform multi-object predictive tracking using Kalman filter for the provided list of STrack instances."""
        if len(stracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks: list[STrack], H: np.ndarray = np.eye(2, 3)):
        """Update state tracks positions and covariances using a homography matrix for multiple tracks."""
        # [ME21] Adjusts the tracks if the camera itself moved (panning/tilting)
        if stracks:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter: KalmanFilterXYAH, frame_id: int):
        """Activate a new tracklet using the provided Kalman filter and initialize its state and covariance."""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False):
        """Reactivate a previously lost track using new detection data and update its state and attributes."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track: STrack, frame_id: int):
        """Update the state of a matched track.

        Args:
            new_track (STrack): The new track containing updated information.
            frame_id (int): The ID of the current frame.

        Examples:
            Update the state of a track with new detection information
            >>> track = STrack([100, 200, 50, 80, 0.9, 1])
            >>> new_track = STrack([105, 205, 55, 85, 0.95, 1])
            >>> track.update(new_track, 2)
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
        """Convert a bounding box's top-left-width-height format to its x-y-aspect-height equivalent."""
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self) -> np.ndarray:
        """Get the bounding box in top-left-width-height format from the current state estimate."""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self) -> np.ndarray:
        """Convert bounding box from (top left x, top left y, width, height) to (min x, min y, max x, max y) format."""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh: np.ndarray) -> np.ndarray:
        """Convert bounding box from tlwh format to center-x-center-y-aspect-height (xyah) format."""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self) -> np.ndarray:
        """Get the current position of the bounding box in (center x, center y, width, height) format."""
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self) -> np.ndarray:
        """Get position in (center x, center y, width, height, angle) format, warning if angle is missing."""
        if self.angle is None:
            LOGGER.warning("`angle` attr not found, returning `xywh` instead.")
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self) -> list[float]:
        """Get the current tracking results in the appropriate bounding box format."""
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]

    def __repr__(self) -> str:
        """Return a string representation of the STrack object including start frame, end frame, and track ID."""
        return f"OT_{self.track_id}_({self.start_frame}-{self.end_frame})"


class BYTETracker:
    """BYTETracker: A tracking algorithm built on top of YOLOv8 for object detection and tracking.

    This class encapsulates the functionality for initializing, updating, and managing the tracks for detected objects
    in a video sequence. It maintains the state of tracked, lost, and removed tracks over frames, utilizes Kalman
    filtering for predicting the new object locations, and performs data association.

    Attributes:
        tracked_stracks (list[STrack]): List of successfully activated tracks.
        lost_stracks (list[STrack]): List of lost tracks.
        removed_stracks (list[STrack]): List of removed tracks.
        frame_id (int): The current frame ID.
        args (Namespace): Command-line arguments.
        max_time_lost (int): The maximum frames for a track to be considered as 'lost'.
        kalman_filter (KalmanFilterXYAH): Kalman Filter object.

    Methods:
        update: Update object tracker with new detections.
        get_kalmanfilter: Return a Kalman filter object for tracking bounding boxes.
        init_track: Initialize object tracking with detections.
        get_dists: Calculate the distance between tracks and detections.
        multi_predict: Predict the location of tracks.
        reset_id: Reset the ID counter of STrack.
        reset: Reset the tracker by clearing all tracks.
        joint_stracks: Combine two lists of stracks.
        sub_stracks: Filter out the stracks present in the second list from the first list.
        remove_duplicate_stracks: Remove duplicate stracks based on IoU.

    Examples:
        Initialize BYTETracker and update with detection results
        >>> tracker = BYTETracker(args, frame_rate=30)
        >>> results = yolo_model.detect(image)
        >>> tracked_objects = tracker.update(results)
    """

    def __init__(self, args, frame_rate: int = 30):
        """Initialize a BYTETracker instance for object tracking.

        Args:
            args (Namespace): Command-line arguments containing tracking parameters.
            frame_rate (int): Frame rate of the video sequence.
        """
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []

        self.frame_id = 0
        self.args = args
        self.max_time_lost = int(frame_rate / 30.0 * args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()

        #--- ME21 : START ---
        # Load args for state file configuration
        self.state_file = getattr(args, 'state_file', "/content") # Path to save/load tracker state
        self.save_interval = getattr(args, 'save_interval', 5000) # Interval to save tracker state

        # Load state if exists
        if self.state_file and os.path.exists(self.state_file):
            LOGGER.info(f"Attempting to load tracker state from {self.state_file}")
            self.load_state(self.state_file)
        else:
            LOGGER.info("No tracker state file found. Create new state file.")
            self.reset_id()
        
        LOGGER.info(f"Modified Tracker Loaded. State file: {self.state_file}, Save interval: {self.save_interval}")
        #--- ME21 : END ---

    def update(self, results, img: np.ndarray | None = None, feats: np.ndarray | None = None) -> np.ndarray:
        """Update the tracker with new detections and return the current list of tracked objects."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = inds_low & inds_high
        results_second = results[inds_second]
        results = results[remain_inds]
        feats_keep = feats_second = img
        if feats is not None and len(feats):
            feats_keep = feats[remain_inds]
            feats_second = feats[inds_second]

        detections = self.init_track(results, feats_keep)
        # Add newly detected tracklets to tracked_stracks
        unconfirmed = []
        tracked_stracks: list[STrack] = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)
        # Step 2: First association, with high score detection boxes
        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        if hasattr(self, "gmc") and img is not None:
            # use try-except here to bypass errors from gmc module
            try:
                warp = self.gmc.apply(img, results.xyxy)
            except Exception:
                warp = np.eye(2, 3)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
        # Step 3: Second association, with low score detection boxes association the untrack to the low score detections
        detections_second = self.init_track(results_second, feats_second)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # TODO: consider fusing scores or appearance features for second association.
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, _u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
        # Deal with unconfirmed tracks, usually tracks with only one beginning frame
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
        # Step 4: Init new stracks
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
        # Step 5: Update state
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-1000:]  # clip removed stracks to 1000 maximum

        #--- ME21 : START ---
        # Auto-save tracker state at specified intervals
        if self.state_file and (self.frame_id % self.save_interval == 0):
            LOGGER.info(f"Auto-saving tracker state at frame {self.frame_id} to {self.state_file}")
            self.save_state(self.state_file)
        #--- ME21 : END ---

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self) -> KalmanFilterXYAH:
        """Return a Kalman filter object for tracking bounding boxes using KalmanFilterXYAH."""
        return KalmanFilterXYAH()

    def init_track(self, results, img: np.ndarray | None = None) -> list[STrack]:
        """Initialize object tracking with given detections, scores, and class labels using the STrack algorithm."""
        if len(results) == 0:
            return []
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        return [STrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks: list[STrack], detections: list[STrack]) -> np.ndarray:
        """Calculate the distance between tracks and detections using IoU and optionally fuse scores."""
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks: list[STrack]):
        """Predict the next states for multiple tracks using Kalman filter."""
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        """Reset the ID counter for STrack instances to ensure unique track IDs across tracking sessions."""
        STrack.reset_id()

    def reset(self):
        """Reset the tracker by clearing all tracked, lost, and removed tracks and reinitializing the Kalman filter."""
        self.tracked_stracks: list[STrack] = []
        self.lost_stracks: list[STrack] = []
        self.removed_stracks: list[STrack] = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    # --- ME21 : START ---
    def save_state(self, filepath: str):
        "Saves the current tracker state to a pickle file."
        LOGGER.info(f"Saving tracker state for Frame {self.frame_id} to {filepath}")

        #Check for valid filepath
        if not self.state_file:
            LOGGER.error("Save failed: self.state_file not specified.")
            return
        
        #Get the next ID from BaseTrack
        """
        BaseTrack (from .basetrack import BaseTrack, TrackState) :
            Base class for object tracking, providing foundational attributes and methods.
        Attributes:
            _count (int): Class-level counter for unique track IDs.
            track_id (int): Unique identifier for the track.
            is_activated (bool): Flag indicating whether the track is currently active.
            state (TrackState): Current state of the track.
            history (OrderedDict): Ordered history of the track's states.
            features (list): List of features extracted from the object for tracking.
            curr_feature (Any): The current feature of the object being tracked.
            score (float): The confidence score of the tracking.
            start_frame (int): The frame number where tracking started.
            frame_id (int): The most recent frame ID processed by the track.
            time_since_update (int): Frames passed since the last update.
            location (tuple): The location of the object in the context of multi-camera tracking.
        """

        next_id = -1
        counter_attr = None
        try:
            # We check multiple names because different YOLO versions 
            # name this counter differently. In our current version, it finds '_count'.
            for candidate in ['_count', 'track_id_count', '_track_id_counter', '_id_count']:
                if hasattr(BaseTrack, candidate):
                    next_id = getattr(BaseTrack, candidate)
                    counter_attr = candidate
                    break

            # Fallback: get max ID from current tracks, preventing duplicates even if we couldn't find the official counter.
            if next_id == -1:
                all_tracks = self.tracked_stracks + self.lost_stracks + self.removed_stracks
                if all_tracks:
                    max_id = max((t.track_id for t in all_tracks if hasattr(t, 'track_id')), default=0)
                    next_id = max_id + 1
                else:
                    next_id = 1
                LOGGER.warning(f"Could not find BaseTrack ID counter. Using max track_id + 1 = {next_id}")
            else:
                LOGGER.info(f"Successfully read {counter_attr}: {next_id}")

        except Exception as e:
            LOGGER.error(f"Could not determine next_id: {e}")
            import traceback
            LOGGER.error(traceback.format_exc()) # copy error (Traceback) to log
            return
        
        state = {
            'frame_id': self.frame_id,
            'next_id': next_id,
            'counter_attr': counter_attr,
            'tracked_stracks': self.tracked_stracks,
            'lost_stracks': self.lost_stracks,
            'removed_stracks': self.removed_stracks[-1000:],  # Only keep recent ones
            'max_time_lost': self.max_time_lost,
        }

        LOGGER.info(f"State bundle created. Preparing {len(state['tracked_stracks'])} tracked + {len(state['lost_stracks'])} lost tracks for pickling...")
        
        # Collect all tracks
        all_tracks = (state['tracked_stracks'] + 
                    state['lost_stracks'] + 
                    state['removed_stracks'])
        
        """
        We implement a custom serialization routine that temporarily detaches the Kalman Filter instances before pickling.
        This avoids TypeError exceptions caused by unpicklable C-extensions within the tracking library.
        A try-finally block ensures these filters are immediately restored, guaranteeing that the runtime memory remains consistent even if the file I/O fails.
        """
        original_filters = {}
        for i, track in enumerate(all_tracks):
            if hasattr(track, 'kalman_filter') and track.kalman_filter is not None:
                original_filters[i] = track.kalman_filter
                track.kalman_filter = None
        try:
            # Ensure directory exists
            dir_path = os.path.dirname(filepath)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            
            # Atomic Save: Write on temporal file and then rename to avoid corruption
            temp_filepath = filepath + ".tmp"
            with open(temp_filepath, 'wb') as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temp_filepath, filepath)
            LOGGER.info(f'--- SUCCESS: Tracker state saved to {filepath} at frame {self.frame_id}. Next ID will be {next_id}. ---')
        
        except Exception as e:
            LOGGER.error(f'--- FAILED to save tracker state to {filepath}: {e} ---')
            import traceback
            LOGGER.error(traceback.format_exc())
        finally:
            # Restore kalman_filter references
            for i, track in enumerate(all_tracks):
                if i in original_filters:
                    track.kalman_filter = original_filters[i]
            LOGGER.info(f"Kalman_filter references restored for {len(all_tracks)} tracks.")

    def load_state(self, filepath: str):
        """Loads the tracker state from a pickle file."""
        try:
            LOGGER.info(f"Loading state from {filepath}...")
            with open(filepath, 'rb') as f:
                state = pickle.load(f)

            # Restore state variables
            self.frame_id = state.get('frame_id', 0)
            self.tracked_stracks = state.get('tracked_stracks', [])
            self.lost_stracks = state.get('lost_stracks', [])
            self.removed_stracks = state.get('removed_stracks', [])
            
            if 'max_time_lost' in state:
                self.max_time_lost = state['max_time_lost']
            
            LOGGER.info(f"Loaded {len(self.tracked_stracks)} tracked + {len(self.lost_stracks)} lost tracks. Re-linking kalman_filter...")

            # Restore kalman_filter reference for all loaded tracks
            for track in self.tracked_stracks + self.lost_stracks:
                track.kalman_filter = self.kalman_filter

            # Update frame_id for lost tracks to give them grace period
            # (end_frame is a property, so we update frame_id instead)
            for track in self.lost_stracks:
                track.frame_id = self.frame_id
                
            # Restore the ID counter
            next_id = state.get('next_id', 1)
            counter_attr = state.get('counter_attr', None)
            
            # Set the ID counter
            counter_set = False
            try:
                candidates = ['_count', 'track_id_count', '_track_id_counter', '_id_count']
                
                # Try the saved counter_attr first
                if counter_attr and hasattr(BaseTrack, counter_attr):
                    setattr(BaseTrack, counter_attr, next_id)
                    counter_set = True
                    LOGGER.info(f"Set BaseTrack.{counter_attr} = {next_id}")
                else:
                    for candidate in candidates:
                        if hasattr(BaseTrack, candidate):
                            setattr(BaseTrack, candidate, next_id)
                            counter_set = True
                            LOGGER.info(f"Set BaseTrack.{candidate} = {next_id}")
                            break
                
                if not counter_set:
                    LOGGER.warning(f"Could not find ID counter attribute in BaseTrack!")
                    LOGGER.warning(f"Available BaseTrack attributes: {[a for a in dir(BaseTrack) if not a.startswith('__')]}")
                    
            except Exception as e:
                LOGGER.error(f"Failed to set ID counter: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
            
            LOGGER.info(f'--- SUCCESS: Tracker state loaded. Resuming from frame {self.frame_id} with next_id {next_id}. ---')
        
        except FileNotFoundError:
            LOGGER.warning(f'--- State file not found: {filepath}. Creating new tracker state. ---')
            self.reset()
        
        except Exception as e:
            LOGGER.error(f'--- FAILED to load tracker state from {filepath}: {e}. Creating new tracker state. ---')
            import traceback
            LOGGER.error(traceback.format_exc())
            self.reset()
    # --- ME21 : END ---
    @staticmethod

    def joint_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Combine two lists of STrack objects into a single list, ensuring no duplicates based on track IDs."""
        exists = {}
        res = []
        for t in tlista:
            exists[t.track_id] = 1
            res.append(t)
        for t in tlistb:
            tid = t.track_id
            if not exists.get(tid, 0):
                exists[tid] = 1
                res.append(t)
        return res

    @staticmethod
    def sub_stracks(tlista: list[STrack], tlistb: list[STrack]) -> list[STrack]:
        """Filter out the stracks present in the second list from the first list."""
        track_ids_b = {t.track_id for t in tlistb}
        return [t for t in tlista if t.track_id not in track_ids_b]

    @staticmethod
    def remove_duplicate_stracks(stracksa: list[STrack], stracksb: list[STrack]) -> tuple[list[STrack], list[STrack]]:
        """Remove duplicate stracks from two lists based on Intersection over Union (IoU) distance."""
        pdist = matching.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb