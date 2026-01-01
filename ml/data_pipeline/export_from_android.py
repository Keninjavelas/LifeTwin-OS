"""
Data export pipeline from Android local database to ML training format.

This script connects the Android data collection system to the ML training pipeline
by exporting data from the encrypted SQLite database to JSON formats suitable for training.
"""

import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AndroidDataExporter:
    """Exports data from Android SQLite database to ML training formats."""
    
    def __init__(self, db_path: str, output_dir: str = "ml/data"):
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def connect_db(self) -> sqlite3.Connection:
        """Connect to the Android SQLite database."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def export_usage_events(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Export usage events for sequence model training."""
        conn = self.connect_db()
        
        # Calculate time range
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT id, packageName, startTime, endTime, totalTimeInForeground, 
               lastTimeUsed, eventType
        FROM usage_events 
        WHERE startTime >= ? AND startTime <= ?
        ORDER BY startTime ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        events = []
        
        for row in cursor:
            events.append({
                "id": row["id"],
                "packageName": row["packageName"],
                "startTime": row["startTime"],
                "endTime": row["endTime"],
                "totalTimeInForeground": row["totalTimeInForeground"],
                "lastTimeUsed": row["lastTimeUsed"],
                "eventType": row["eventType"],
                "timestamp": row["startTime"]  # For compatibility
            })
        
        conn.close()
        logger.info(f"Exported {len(events)} usage events")
        return events
    
    def export_notification_events(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Export notification events."""
        conn = self.connect_db()
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT id, packageName, timestamp, category, priority, hasActions, isOngoing
        FROM notification_events 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        events = []
        
        for row in cursor:
            events.append({
                "id": row["id"],
                "packageName": row["packageName"],
                "timestamp": row["timestamp"],
                "category": row["category"],
                "priority": row["priority"],
                "hasActions": bool(row["hasActions"]),
                "isOngoing": bool(row["isOngoing"])
            })
        
        conn.close()
        logger.info(f"Exported {len(events)} notification events")
        return events
    
    def export_screen_sessions(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Export screen session data."""
        conn = self.connect_db()
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT sessionId, startTime, endTime, unlockCount, interactionIntensity
        FROM screen_sessions 
        WHERE startTime >= ? AND startTime <= ?
        ORDER BY startTime ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        sessions = []
        
        for row in cursor:
            duration = (row["endTime"] or end_time) - row["startTime"]
            sessions.append({
                "sessionId": row["sessionId"],
                "startTime": row["startTime"],
                "endTime": row["endTime"],
                "duration": duration,
                "unlockCount": row["unlockCount"],
                "interactionIntensity": row["interactionIntensity"]
            })
        
        conn.close()
        logger.info(f"Exported {len(sessions)} screen sessions")
        return sessions
    
    def export_interaction_metrics(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Export interaction metrics."""
        conn = self.connect_db()
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT id, timestamp, touchCount, scrollEvents, gesturePatterns, 
               interactionIntensity, timeWindowStart, timeWindowEnd
        FROM interaction_metrics 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        metrics = []
        
        for row in cursor:
            # Parse gesture patterns JSON
            gesture_patterns = []
            try:
                if row["gesturePatterns"]:
                    gesture_patterns = json.loads(row["gesturePatterns"])
            except (json.JSONDecodeError, TypeError):
                pass
            
            metrics.append({
                "id": row["id"],
                "timestamp": row["timestamp"],
                "touchCount": row["touchCount"],
                "scrollEvents": row["scrollEvents"],
                "gesturePatterns": gesture_patterns,
                "interactionIntensity": row["interactionIntensity"],
                "timeWindow": {
                    "startTime": row["timeWindowStart"],
                    "endTime": row["timeWindowEnd"]
                }
            })
        
        conn.close()
        logger.info(f"Exported {len(metrics)} interaction metrics")
        return metrics
    
    def export_activity_contexts(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Export sensor activity contexts."""
        conn = self.connect_db()
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT id, activityType, confidence, timestamp, duration
        FROM activity_contexts 
        WHERE timestamp >= ? AND timestamp <= ?
        ORDER BY timestamp ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        contexts = []
        
        for row in cursor:
            contexts.append({
                "id": row["id"],
                "activityType": row["activityType"],
                "confidence": row["confidence"],
                "timestamp": row["timestamp"],
                "duration": row["duration"]
            })
        
        conn.close()
        logger.info(f"Exported {len(contexts)} activity contexts")
        return contexts
    
    def export_daily_summaries(self, days_back: int = 90) -> List[Dict[str, Any]]:
        """Export daily summaries for time-series training."""
        conn = self.connect_db()
        
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp() * 1000)
        
        query = """
        SELECT id, date, totalScreenTime, totalUnlocks, topApps, 
               averageSessionDuration, notificationCount, interactionIntensity,
               dominantActivity, energyLevel, focusLevel, moodLevel
        FROM daily_summaries 
        WHERE date >= ? AND date <= ?
        ORDER BY date ASC
        """
        
        cursor = conn.execute(query, (start_time, end_time))
        summaries = []
        
        for row in cursor:
            # Parse top apps JSON
            top_apps = []
            try:
                if row["topApps"]:
                    top_apps = json.loads(row["topApps"])
            except (json.JSONDecodeError, TypeError):
                pass
            
            summaries.append({
                "id": row["id"],
                "timestamp": datetime.fromtimestamp(row["date"] / 1000).isoformat() + "Z",
                "date": row["date"],
                "total_screen_time": row["totalScreenTime"],
                "total_unlocks": row["totalUnlocks"],
                "top_apps": top_apps,
                "average_session_duration": row["averageSessionDuration"],
                "notification_count": row["notificationCount"],
                "interaction_intensity": row["interactionIntensity"],
                "dominant_activity": row["dominantActivity"],
                "energy_level": row["energyLevel"],
                "focus_level": row["focusLevel"],
                "mood_level": row["moodLevel"],
                # Legacy compatibility
                "most_common_hour": 14,  # Placeholder
                "predicted_next_app": None
            })
        
        conn.close()
        logger.info(f"Exported {len(summaries)} daily summaries")
        return summaries
    
    def create_app_sequences(self, usage_events: List[Dict[str, Any]], 
                           gap_minutes: int = 5) -> List[List[str]]:
        """Create app usage sequences for next-app prediction training."""
        if not usage_events:
            return []
        
        # Sort events by timestamp
        events = sorted(usage_events, key=lambda x: x["startTime"])
        
        sequences = []
        current_sequence = []
        last_time = None
        gap_ms = gap_minutes * 60 * 1000
        
        for event in events:
            current_time = event["startTime"]
            package_name = event["packageName"]
            
            # Skip system apps and very short sessions
            if (package_name.startswith("com.android.") or 
                package_name.startswith("com.google.android.") or
                event.get("totalTimeInForeground", 0) < 5000):  # Less than 5 seconds
                continue
            
            if last_time is None or (current_time - last_time) > gap_ms:
                # Start new sequence
                if len(current_sequence) >= 2:  # Only keep sequences with 2+ apps
                    sequences.append(current_sequence)
                current_sequence = [package_name]
            else:
                # Continue current sequence
                if package_name != current_sequence[-1]:  # Avoid consecutive duplicates
                    current_sequence.append(package_name)
            
            last_time = current_time
        
        # Add final sequence
        if len(current_sequence) >= 2:
            sequences.append(current_sequence)
        
        logger.info(f"Created {len(sequences)} app usage sequences")
        return sequences
    
    def export_for_sequence_training(self, days_back: int = 30) -> Dict[str, Any]:
        """Export data formatted for sequence model training."""
        usage_events = self.export_usage_events(days_back)
        sequences = self.create_app_sequences(usage_events)
        
        # Create app frequency mapping for vocabulary
        app_counts = {}
        for sequence in sequences:
            for app in sequence:
                app_counts[app] = app_counts.get(app, 0) + 1
        
        # Filter to most common apps (vocabulary size limit)
        min_occurrences = 3
        vocab_apps = {app: count for app, count in app_counts.items() 
                     if count >= min_occurrences}
        
        # Filter sequences to only include vocab apps
        filtered_sequences = []
        for sequence in sequences:
            filtered_seq = [app for app in sequence if app in vocab_apps]
            if len(filtered_seq) >= 2:
                filtered_sequences.append(filtered_seq)
        
        export_data = {
            "sequences": filtered_sequences,
            "vocabulary": vocab_apps,
            "total_events": len(usage_events),
            "total_sequences": len(sequences),
            "filtered_sequences": len(filtered_sequences),
            "vocab_size": len(vocab_apps),
            "export_timestamp": datetime.now().isoformat() + "Z"
        }
        
        return export_data
    
    def export_for_timeseries_training(self, days_back: int = 90) -> Dict[str, Any]:
        """Export data formatted for time-series model training."""
        summaries = self.export_daily_summaries(days_back)
        
        # Create time series data
        time_series = {
            "screen_time": [s["total_screen_time"] for s in summaries],
            "unlock_count": [s["total_unlocks"] for s in summaries],
            "notification_count": [s["notification_count"] for s in summaries],
            "interaction_intensity": [s["interaction_intensity"] for s in summaries],
            "energy_level": [s.get("energy_level", 0.5) for s in summaries],
            "focus_level": [s.get("focus_level", 0.5) for s in summaries],
            "mood_level": [s.get("mood_level", 0.5) for s in summaries],
            "dates": [s["timestamp"] for s in summaries]
        }
        
        export_data = {
            "summaries": summaries,  # For compatibility with existing training code
            "time_series": time_series,
            "total_days": len(summaries),
            "date_range": {
                "start": summaries[0]["timestamp"] if summaries else None,
                "end": summaries[-1]["timestamp"] if summaries else None
            },
            "export_timestamp": datetime.now().isoformat() + "Z"
        }
        
        return export_data
    
    def export_all_data(self, days_back: int = 30) -> Dict[str, Any]:
        """Export all data types for comprehensive ML training."""
        logger.info("Starting comprehensive data export...")
        
        export_data = {
            "device_id": "android_device",
            "export_timestamp": datetime.now().isoformat() + "Z",
            "days_back": days_back,
            
            # Raw events
            "usage_events": self.export_usage_events(days_back),
            "notification_events": self.export_notification_events(days_back),
            "screen_sessions": self.export_screen_sessions(days_back),
            "interaction_metrics": self.export_interaction_metrics(days_back),
            "activity_contexts": self.export_activity_contexts(days_back),
            
            # Processed summaries
            "daily_summaries": self.export_daily_summaries(min(days_back * 3, 90)),
            
            # ML-ready formats
            "sequence_data": self.export_for_sequence_training(days_back),
            "timeseries_data": self.export_for_timeseries_training(min(days_back * 3, 90))
        }
        
        logger.info("Data export completed successfully")
        return export_data
    
    def save_exports(self, export_data: Dict[str, Any]):
        """Save exported data to various formats for different training scripts."""
        
        # Save comprehensive export
        comprehensive_path = self.output_dir / "comprehensive_export.json"
        with open(comprehensive_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comprehensive export to {comprehensive_path}")
        
        # Save sequence training format (compatible with existing training script)
        sequence_export = {
            "device_id": export_data["device_id"],
            "summaries": export_data["daily_summaries"],
            "sequences": export_data["sequence_data"]["sequences"],
            "vocabulary": export_data["sequence_data"]["vocabulary"]
        }
        
        sequence_path = self.output_dir / "summaries_export.json"
        with open(sequence_path, 'w', encoding='utf-8') as f:
            json.dump(sequence_export, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved sequence training data to {sequence_path}")
        
        # Save time-series training format
        timeseries_path = self.output_dir / "timeseries_export.json"
        with open(timeseries_path, 'w', encoding='utf-8') as f:
            json.dump(export_data["timeseries_data"], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved time-series training data to {timeseries_path}")
        
        # Save raw events in JSONL format for preprocessing
        events_path = self.output_dir / "events.jsonl"
        with open(events_path, 'w', encoding='utf-8') as f:
            for event in export_data["usage_events"]:
                f.write(json.dumps(event) + '\n')
        logger.info(f"Saved raw events to {events_path}")
        
        # Save sequences in JSONL format
        sequences_path = self.output_dir / "sequences.jsonl"
        with open(sequences_path, 'w', encoding='utf-8') as f:
            for sequence in export_data["sequence_data"]["sequences"]:
                f.write(json.dumps(sequence) + '\n')
        logger.info(f"Saved sequences to {sequences_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Android data for ML training")
    parser.add_argument("--db-path", required=True, 
                       help="Path to Android SQLite database")
    parser.add_argument("--output-dir", default="ml/data",
                       help="Output directory for exported data")
    parser.add_argument("--days-back", type=int, default=30,
                       help="Number of days to export")
    parser.add_argument("--export-type", choices=["all", "sequence", "timeseries"], 
                       default="all", help="Type of export to perform")
    
    args = parser.parse_args()
    
    try:
        exporter = AndroidDataExporter(args.db_path, args.output_dir)
        
        if args.export_type == "sequence":
            data = exporter.export_for_sequence_training(args.days_back)
            output_path = Path(args.output_dir) / "sequence_export.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Sequence data exported to {output_path}")
            
        elif args.export_type == "timeseries":
            data = exporter.export_for_timeseries_training(args.days_back)
            output_path = Path(args.output_dir) / "timeseries_export.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Time-series data exported to {output_path}")
            
        else:  # all
            data = exporter.export_all_data(args.days_back)
            exporter.save_exports(data)
        
        logger.info("Export completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()