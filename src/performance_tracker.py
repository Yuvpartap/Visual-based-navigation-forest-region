"""
Performance tracking utilities for monitoring processing speed and timing.
"""

import numpy as np
from collections import deque


class PerformanceTracker:
    """Tracks performance metrics for video processing."""
    
    def __init__(self):
        """Initialize performance tracker with empty statistics."""
        self.stats = {'total': [], 'matching': [], 'homography': []}
        self.frame_times = deque(maxlen=10)
    
    def record(self, category, duration):
        """
        Record a timing measurement.
        
        Args:
            category: Category name ('total', 'matching', 'homography')
            duration: Duration in seconds
        """
        if category in self.stats:
            self.stats[category].append(duration)
    
    def record_frame_time(self, duration):
        """
        Record total frame processing time.
        
        Args:
            duration: Frame processing duration in seconds
        """
        self.frame_times.append(duration)
    
    def get_average(self, category='total'):
        """
        Get average time for a category.
        
        Args:
            category: Category name
            
        Returns:
            Average time in seconds
        """
        if not self.stats.get(category):
            return 0.0
        return np.mean(self.stats[category])
    
    def get_breakdown(self):
        """
        Get timing breakdown for all categories.
        
        Returns:
            Dictionary with average times for each category
        """
        return {
            'total': self.get_average('total'),
            'matching': self.get_average('matching'),
            'homography': self.get_average('homography'),
        }
    
    def get_recent_fps(self):
        """
        Calculate recent FPS based on frame times.
        
        Returns:
            FPS (frames per second)
        """
        if not self.frame_times:
            return 0.0
        return 1.0 / np.mean(self.frame_times)
