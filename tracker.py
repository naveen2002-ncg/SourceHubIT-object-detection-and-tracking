import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from utils import calculate_iou
import time

class Track:
    """Represents a single tracked object."""
    
    def __init__(self, bbox: List[float], class_id: int, confidence: float, track_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.track_id = track_id
        self.age = 0
        self.total_hits = 1
        self.time_since_update = 0
        self.last_update = time.time()
        
        # Kalman filter for prediction
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        
        # Initialize Kalman filter with center point
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        self.kalman.statePre = np.array([[center_x], [center_y], [0], [0]], np.float32)
        self.kalman.statePost = np.array([[center_x], [center_y], [0], [0]], np.float32)
    
    def predict(self) -> List[float]:
        """Predict next position using Kalman filter."""
        prediction = self.kalman.predict()
        center_x, center_y = prediction[0][0], prediction[1][0]
        
        # Estimate bbox size from previous measurements
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        
        x1 = center_x - width / 2
        y1 = center_y - height / 2
        x2 = center_x + width / 2
        y2 = center_y + height / 2
        
        return [x1, y1, x2, y2]
    
    def update(self, bbox: List[float], confidence: float):
        """Update track with new detection."""
        self.bbox = bbox
        self.confidence = confidence
        self.total_hits += 1
        self.time_since_update = 0
        self.last_update = time.time()
        
        # Update Kalman filter
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        measurement = np.array([[center_x], [center_y]], np.float32)
        self.kalman.correct(measurement)
    
    def mark_missed(self):
        """Mark track as missed in current frame."""
        self.time_since_update += 1
        self.age += 1

class ObjectTracker:
    """Multi-object tracker using IoU-based association."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize the tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without updates
            min_hits: Minimum number of detections before a track is confirmed
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detection dictionaries with 'bbox', 'class_id', 'confidence' keys
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to existing tracks
        matched_detections, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matched_detections:
            self.tracks[track_idx].update(
                detections[detection_idx]['bbox'],
                detections[detection_idx]['confidence']
            )
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_track = Track(
                detection['bbox'],
                detection['class_id'],
                detection['confidence'],
                self.next_id
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]
        
        # Return only confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.total_hits >= self.min_hits:
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'class_id': track.class_id,
                    'confidence': track.confidence,
                    'age': track.age,
                    'total_hits': track.total_hits
                })
        
        return confirmed_tracks
    
    def _associate_detections_to_tracks(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing tracks using IoU.
        
        Args:
            detections: List of detection dictionaries
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(track.bbox, detection['bbox'])
        
        # Use Hungarian algorithm for optimal assignment
        from scipy.optimize import linear_sum_assignment
        track_indices, detection_indices = linear_sum_assignment(-iou_matrix)
        
        matched_pairs = []
        unmatched_detections = []
        unmatched_tracks = []
        
        # Check IoU threshold and create matched pairs
        for track_idx, detection_idx in zip(track_indices, detection_indices):
            if iou_matrix[track_idx, detection_idx] >= self.iou_threshold:
                matched_pairs.append((track_idx, detection_idx))
            else:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
        
        # Find unmatched tracks and detections
        matched_track_indices = [pair[0] for pair in matched_pairs]
        matched_detection_indices = [pair[1] for pair in matched_pairs]
        
        for i in range(len(self.tracks)):
            if i not in matched_track_indices:
                unmatched_tracks.append(i)
        
        for i in range(len(detections)):
            if i not in matched_detection_indices:
                unmatched_detections.append(i)
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len([track for track in self.tracks if track.total_hits >= self.min_hits])
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1

class SimpleTracker:
    """Simplified tracker for cases where scipy is not available."""
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.frame_count = 0
        self.next_id = 1
    
    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracker with new detections using greedy association."""
        self.frame_count += 1
        
        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to existing tracks using greedy approach
        matched_detections, unmatched_detections, unmatched_tracks = self._greedy_association(detections)
        
        # Update matched tracks
        for track_idx, detection_idx in matched_detections:
            self.tracks[track_idx].update(
                detections[detection_idx]['bbox'],
                detections[detection_idx]['confidence']
            )
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            new_track = Track(
                detection['bbox'],
                detection['class_id'],
                detection['confidence'],
                self.next_id
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [track for track in self.tracks if track.time_since_update < self.max_age]
        
        # Return only confirmed tracks
        confirmed_tracks = []
        for track in self.tracks:
            if track.total_hits >= self.min_hits:
                confirmed_tracks.append({
                    'track_id': track.track_id,
                    'bbox': track.bbox,
                    'class_id': track.class_id,
                    'confidence': track.confidence,
                    'age': track.age,
                    'total_hits': track.total_hits
                })
        
        return confirmed_tracks
    
    def _greedy_association(self, detections: List[Dict[str, Any]]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Greedy association of detections to tracks."""
        if len(self.tracks) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(self.tracks)))
        
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                iou_matrix[i, j] = calculate_iou(track.bbox, detection['bbox'])
        
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        # Greedy assignment
        while True:
            max_iou = 0
            best_track_idx = -1
            best_detection_idx = -1
            
            for track_idx in unmatched_tracks:
                for detection_idx in unmatched_detections:
                    if iou_matrix[track_idx, detection_idx] > max_iou:
                        max_iou = iou_matrix[track_idx, detection_idx]
                        best_track_idx = track_idx
                        best_detection_idx = detection_idx
            
            if max_iou >= self.iou_threshold:
                matched_pairs.append((best_track_idx, best_detection_idx))
                unmatched_tracks.remove(best_track_idx)
                unmatched_detections.remove(best_detection_idx)
            else:
                break
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len([track for track in self.tracks if track.total_hits >= self.min_hits])
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = []
        self.frame_count = 0
        self.next_id = 1 