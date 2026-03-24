import json
import math
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Tuple
import uuid


class ProcessingMode(Enum):
    LAZY = "lazy"
    EAGER = "eager"
    HYBRID = "hybrid"


class AnomalySeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyReport:
    timestamp: datetime
    severity: AnomalySeverity
    anomaly_type: str
    description: str
    context: Dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class WindowStatistics:
    start_index: int
    end_index: int
    duration_seconds: float
    event_counts: Dict[str, int]
    entropy: float
    sample_events: List[str]
    anomaly_flags: List[AnomalyReport]


class SlidingWindowEntropy:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)
        self._counter: Counter = Counter()
        self._total_count = 0
        self._current_entropy = 0.0
    
    def update(self, value: Any) -> None:
        if len(self._buffer) == self.window_size:
            oldest = self._buffer[0]
            old_count = self._counter[oldest]
            self._counter[oldest] = old_count - 1
            if self._counter[oldest] == 0:
                del self._counter[oldest]
            self._total_count -= 1
        
        self._buffer.append(value)
        self._counter[value] += 1
        self._total_count += 1
        self._recalculate_entropy()
    
    def _recalculate_entropy(self) -> None:
        if self._total_count == 0:
            self._current_entropy = 0.0
            return
        
        entropy = 0.0
        for count in self._counter.values():
            probability = count / self._total_count
            if probability > 0:
                entropy -= probability * math.log2(probability)
        
        self._current_entropy = entropy
    
    @property
    def entropy(self) -> float:
        return self._current_entropy
    
    @property
    def unique_values(self) -> int:
        return len(self._counter)
    
    def reset(self) -> None:
        self._buffer.clear()
        self._counter.clear()
        self._total_count = 0
        self._current_entropy = 0.0


class ExponentialSmoothingEntropy:
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor
        self._current_entropy = 0.0
        self._initialized = False
    
    def update(self, raw_entropy: float) -> None:
        if not self._initialized:
            self._current_entropy = raw_entropy
            self._initialized = True
        else:
            self._current_entropy = (self._current_entropy * self.decay_factor) + (raw_entropy * (1 - self.decay_factor))
    
    @property
    def entropy(self) -> float:
        return self._current_entropy
    
    def reset(self) -> None:
        self._current_entropy = 0.0
        self._initialized = False


class PatternDetector:
    def __init__(self):
        self._patterns: Dict[str, re.Pattern] = {}
        self._pattern_stats: Dict[str, Counter] = {}
    
    def register_pattern(self, name: str, pattern: str, flags: int = re.IGNORECASE) -> None:
        self._patterns[name] = re.compile(pattern, flags)
        self._pattern_stats[name] = Counter()
    
    def scan_line(self, line: str) -> Dict[str, bool]:
        results = {}
        for name, pattern in self._patterns.items():
            match = pattern.search(line)
            results[name] = match is not None
            if match:
                self._pattern_stats[name][match.group(0)] += 1
        return results
    
    def get_frequent_matches(self, pattern_name: str, limit: int = 10) -> List[Tuple[str, int]]:
        if pattern_name not in self._pattern_stats:
            return []
        return self._pattern_stats[pattern_name].most_common(limit)
    
    def get_pattern_statistics(self) -> Dict[str, Dict[str, Any]]:
        stats = {}
        for name, counter in self._pattern_stats.items():
            total_matches = sum(counter.values())
            unique_matches = len(counter)
            stats[name] = {
                'total_matches': total_matches,
                'unique_matches': unique_matches,
                'most_common': counter.most_common(5),
                'diversity_ratio': unique_matches / total_matches if total_matches > 0 else 0
            }
        return stats


class AdaptiveThresholdMonitor:
    def __init__(self, baseline_window: int = 100, sensitivity: float = 2.0):
        self.baseline_window = baseline_window
        self.sensitivity = sensitivity
        self._values: deque = deque(maxlen=baseline_window * 2)
        self._baseline_mean = 0.0
        self._baseline_variance = 0.0
        self._baseline_std = 0.0
        self._is_calibrated = False
    
    def test_anomaly(self, value: float) -> Tuple[bool, float]:
        if not self._is_calibrated:
            return False, 0.0
        
        deviation = abs(value - self._baseline_mean)
        z_score = deviation / self._baseline_std if self._baseline_std > 0 else 0.0
        is_anomalous = z_score > self.sensitivity
        
        return is_anomalous, z_score
    
    def update_baseline(self, value: float) -> None:
        self._values.append(value)
        
        if len(self._values) >= self.baseline_window and not self._is_calibrated:
            self._calibrate()
        elif self._is_calibrated:
            self._update_baseline_ewma(value)
    
    def _calibrate(self) -> None:
        recent = list(self._values)[-self.baseline_window:]
        self._baseline_mean = sum(recent) / len(recent)
        variance = sum((x - self._baseline_mean) ** 2 for x in recent) / len(recent)
        self._baseline_variance = variance
        self._baseline_std = math.sqrt(variance)
        self._is_calibrated = True
    
    def _update_baseline_ewma(self, value: float) -> None:
        alpha = 0.01
        self._baseline_mean = (1 - alpha) * self._baseline_mean + alpha * value
        delta = value - self._baseline_mean
        self._baseline_variance = (1 - alpha) * self._baseline_variance + alpha * (delta ** 2)
        self._baseline_std = math.sqrt(self._baseline_variance)
    
    @property
    def baseline(self) -> Tuple[float, float]:
        return self._baseline_mean, self._baseline_std
    
    @property
    def calibrated(self) -> bool:
        return self._is_calibrated
    
    def reset(self) -> None:
        self._values.clear()
        self._baseline_mean = 0.0
        self._baseline_variance = 0.0
        self._baseline_std = 0.0
        self._is_calibrated = False


class TemporalPatternAnalyzer:
    def __init__(self, time_window_seconds: int = 3600):
        self.time_window = time_window_seconds
        self._timestamps: deque = deque()
        self._values: deque = deque()
    
    def record(self, timestamp: float, value: float) -> None:
        self._timestamps.append(timestamp)
        self._values.append(value)
        
        cutoff = timestamp - self.time_window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._values.popleft()
    
    def detect_arrival_regularity(self, min_occurrences: int = 3) -> Dict[str, Any]:
        if len(self._timestamps) < min_occurrences:
            return {'regular': False, 'reason': 'insufficient data'}
        
        intervals = []
        for i in range(1, len(self._timestamps)):
            interval = self._timestamps[i] - self._timestamps[i-1]
            intervals.append(interval)
        
        if not intervals:
            return {'regular': False, 'reason': 'no intervals'}
        
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        coefficient_of_variation = std_dev / avg_interval if avg_interval > 0 else 1.0
        regularity = 1.0 - min(1.0, coefficient_of_variation)
        
        is_regular = regularity > 0.7 and len(intervals) >= min_occurrences
        
        return {
            'regular': is_regular,
            'average_interval': avg_interval,
            'regularity_score': regularity,
            'observed_intervals': len(intervals),
            'interval_std_dev': std_dev,
            'coefficient_of_variation': coefficient_of_variation
        }
    
    def get_trend(self) -> Dict[str, Any]:
        if len(self._values) < 2:
            return {'trend_detected': False}
        
        n = len(self._values)
        x_sum = sum(range(n))
        y_sum = sum(self._values)
        xy_sum = sum(i * v for i, v in enumerate(self._values))
        x2_sum = sum(i * i for i in range(n))
        
        denominator = (n * x2_sum - x_sum * x_sum)
        if denominator == 0:
            return {'trend_detected': False}
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        intercept = (y_sum - slope * x_sum) / n
        
        predicted = [slope * i + intercept for i in range(n)]
        residuals = [self._values[i] - predicted[i] for i in range(n)]
        mse = sum(r * r for r in residuals) / n
        rmse = math.sqrt(mse)
        
        if slope > 0:
            direction = "increasing"
        elif slope < 0:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            'trend_detected': True,
            'slope': slope,
            'intercept': intercept,
            'rmse': rmse,
            'direction': direction,
            'strength': min(1.0, abs(slope) / (abs(slope) + rmse)) if rmse > 0 else 1.0
        }


class AdvancedLogProcessor:
    def __init__(
        self,
        correlation_window_seconds: int = 60,
        event_buffer_size: int = 10000,
        processing_mode: ProcessingMode = ProcessingMode.HYBRID
    ):
        self.correlation_window = correlation_window_seconds
        self.processing_mode = processing_mode
        self._pattern_detector = PatternDetector()
        self._event_distribution_entropy = SlidingWindowEntropy(window_size=500)
        self._smoothed_entropy = ExponentialSmoothingEntropy(decay_factor=0.95)
        self._threshold_monitor = AdaptiveThresholdMonitor(baseline_window=200, sensitivity=2.5)
        self._temporal_analyzer = TemporalPatternAnalyzer(time_window_seconds=3600)
        self._event_buffer: deque = deque(maxlen=event_buffer_size)
        self._correlation_groups: Dict[str, List[Dict[str, Any]]] = {}
        self._anomaly_history: List[AnomalyReport] = []
        
        self._pattern_detector.register_pattern('error', r'ERROR|FATAL|CRITICAL')
        self._pattern_detector.register_pattern('warning', r'WARNING|WARN')
        self._pattern_detector.register_pattern('exception', r'Exception|Traceback|Stack trace')
        self._pattern_detector.register_pattern('timeout', r'timeout|timed out')
        self._pattern_detector.register_pattern('connection', r'connection|disconnect|reconnect')
        self._pattern_detector.register_pattern('authentication', r'auth|login|password|token')
        self._pattern_detector.register_pattern('database', r'database|db|sql|query')
        self._pattern_detector.register_pattern('memory', r'memory|heap|allocation')
    
    def process_line(self, line: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()
        
        line_length = len(line)
        word_count = len(line.split())
        pattern_matches = self._pattern_detector.scan_line(line)
        
        event_signature = self._extract_event_signature(line, pattern_matches)
        self._event_distribution_entropy.update(event_signature)
        raw_entropy = self._event_distribution_entropy.entropy
        self._smoothed_entropy.update(raw_entropy)
        
        self._temporal_analyzer.record(timestamp, line_length)
        
        anomaly, z_score = self._threshold_monitor.test_anomaly(line_length)
        
        self._threshold_monitor.update_baseline(line_length)
        
        event = {
            'timestamp': timestamp,
            'line': line,
            'length': line_length,
            'word_count': word_count,
            'patterns': pattern_matches,
            'raw_entropy': raw_entropy,
            'smoothed_entropy': self._smoothed_entropy.entropy,
            'is_anomaly': anomaly,
            'anomaly_score': z_score,
            'correlation_id': None
        }
        
        self._event_buffer.append(event)
        
        if anomaly:
            correlation_id = self._correlate_event(event)
            event['correlation_id'] = correlation_id
            
            anomaly_report = AnomalyReport(
                timestamp=datetime.fromtimestamp(timestamp),
                severity=AnomalySeverity.MEDIUM if z_score > 3 else AnomalySeverity.LOW,
                anomaly_type='length_anomaly',
                description=f"Line length {line_length} deviates from baseline (z-score: {z_score:.2f})",
                context={
                    'line_preview': line[:200],
                    'z_score': z_score,
                    'matched_patterns': [p for p, m in pattern_matches.items() if m],
                    'line_length': line_length,
                    'correlation_id': correlation_id
                },
                correlation_id=correlation_id
            )
            self._anomaly_history.append(anomaly_report)
        
        return event
    
    def _extract_event_signature(self, line: str, pattern_matches: Dict[str, bool]) -> str:
        matched_patterns = [p for p, matched in pattern_matches.items() if matched]
        if matched_patterns:
            return f"pattern:{','.join(matched_patterns)}"
        
        words = line.split()
        if len(words) >= 2:
            return f"type:{words[0] if words[0].startswith('[') else words[1] if len(words) > 1 else words[0]}"
        
        return "type:unknown"
    
    def _correlate_event(self, event: Dict[str, Any]) -> str:
        current_time = event['timestamp']
        patterns = [p for p, matched in event['patterns'].items() if matched]
        
        if not patterns:
            return str(uuid.uuid4())
        
        correlation_key = patterns[0]
        cutoff = current_time - self.correlation_window
        
        if correlation_key in self._correlation_groups:
            self._correlation_groups[correlation_key] = [
                e for e in self._correlation_groups[correlation_key]
                if e['timestamp'] > cutoff
            ]
            
            if self._correlation_groups[correlation_key]:
                return self._correlation_groups[correlation_key][0]['correlation_id']
        
        correlation_id = str(uuid.uuid4())
        self._correlation_groups.setdefault(correlation_key, []).append({
            'timestamp': current_time,
            'correlation_id': correlation_id,
            'patterns': patterns
        })
        
        return correlation_id
    
    def get_window_statistics(self, window_seconds: int = 300) -> WindowStatistics:
        cutoff = time.time() - window_seconds
        window_events = [e for e in self._event_buffer if e['timestamp'] > cutoff]
        
        if not window_events:
            return WindowStatistics(
                start_index=0,
                end_index=0,
                duration_seconds=0,
                event_counts={},
                entropy=0.0,
                sample_events=[],
                anomaly_flags=[]
            )
        
        event_counts = Counter()
        window_anomalies = []
        
        for event in window_events:
            for pattern, matched in event['patterns'].items():
                if matched:
                    event_counts[pattern] += 1
            
            if event['is_anomaly']:
                window_anomalies.append(AnomalyReport(
                    timestamp=datetime.fromtimestamp(event['timestamp']),
                    severity=AnomalySeverity.MEDIUM if event['anomaly_score'] > 3 else AnomalySeverity.LOW,
                    anomaly_type='length_anomaly',
                    description=f"Line length {event['length']} deviates from baseline",
                    context={
                        'line_preview': event['line'][:200],
                        'z_score': event['anomaly_score'],
                        'patterns': event['patterns']
                    },
                    correlation_id=event.get('correlation_id', str(uuid.uuid4()))
                ))
        
        sample_lines = [e['line'][:150] for e in window_events[-5:]]
        
        return WindowStatistics(
            start_index=len(self._event_buffer) - len(window_events),
            end_index=len(self._event_buffer),
            duration_seconds=window_events[-1]['timestamp'] - window_events[0]['timestamp'],
            event_counts=dict(event_counts),
            entropy=self._event_distribution_entropy.entropy,
            sample_events=sample_lines,
            anomaly_flags=window_anomalies
        )
    
    def get_anomaly_history(self, limit: int = 100) -> List[AnomalyReport]:
        return self._anomaly_history[-limit:]
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        return {
            'pattern_statistics': self._pattern_detector.get_pattern_statistics(),
            'entropy_state': {
                'current_raw_entropy': self._event_distribution_entropy.entropy,
                'current_smoothed_entropy': self._smoothed_entropy.entropy,
                'unique_event_signatures': self._event_distribution_entropy.unique_values,
                'window_size': self._event_distribution_entropy.window_size
            },
            'threshold_baseline': {
                'mean': self._threshold_monitor.baseline[0],
                'std': self._threshold_monitor.baseline[1],
                'calibrated': self._threshold_monitor.calibrated
            },
            'temporal_analysis': {
                'arrival_regularity': self._temporal_analyzer.detect_arrival_regularity(),
                'trend': self._temporal_analyzer.get_trend()
            },
            'correlation_groups': {
                pattern: len(events) for pattern, events in self._correlation_groups.items()
            },
            'total_events_processed': len(self._event_buffer),
            'total_anomalies_detected': len(self._anomaly_history),
            'processing_mode': self.processing_mode.value
        }
    
    def reset(self) -> None:
        self._event_distribution_entropy.reset()
        self._smoothed_entropy.reset()
        self._threshold_monitor.reset()
        self._event_buffer.clear()
        self._correlation_groups.clear()
        self._anomaly_history.clear()


class TimestampSimulator:
    def __init__(self, start_time: float, speed_factor: float = 1.0):
        self._base_time = start_time
        self._current_time = start_time
        self._speed_factor = speed_factor
        self._event_counter = 0
    
    def generate_timestamp(self, offset_seconds: float) -> float:
        self._current_time = self._base_time + (offset_seconds / self._speed_factor)
        self._event_counter += 1
        return self._current_time
    
    def reset(self, new_start_time: Optional[float] = None) -> None:
        if new_start_time is None:
            new_start_time = time.time()
        self._base_time = new_start_time
        self._current_time = new_start_time
        self._event_counter = 0
    
    @property
    def events_generated(self) -> int:
        return self._event_counter


class DataStreamSimulator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self._event_types = [
            'INFO: User action completed',
            'WARNING: Slow query detected',
            'ERROR: Connection pool exhausted',
            'INFO: Cache hit',
            'WARNING: Memory usage above threshold',
            'ERROR: Authentication failure',
            'INFO: Background task started',
            'CRITICAL: Service unreachable',
            'DEBUG: Processing request',
            'ERROR: Database timeout'
        ]
        
        self._base_rate = 1.0
        self._burst_mode = False
        self._burst_end_offset = 0
    
    def _poisson(self, lam: float) -> int:
        if lam < 100:
            p = math.exp(-lam)
            k = 0
            s = p
            u = random.random()
            while u > s:
                k += 1
                p *= lam / k
                s += p
            return k
        else:
            return int(random.gauss(lam, math.sqrt(lam)) + 0.5)
    
    def generate_events(
        self,
        duration_seconds: int = 30,
        events_per_second: int = 10,
        timestamp_simulator: Optional[TimestampSimulator] = None
    ) -> Generator[Tuple[float, str], None, None]:
        start_offset = 0
        end_offset = duration_seconds
        current_offset = 0
        event_counter = 0
        
        use_simulator = timestamp_simulator is not None
        base_time = timestamp_simulator._base_time if use_simulator else time.time()
        
        while current_offset < end_offset:
            if self._burst_mode and current_offset > self._burst_end_offset:
                self._burst_mode = False
            
            rate = self._base_rate * (5 if self._burst_mode else 1)
            events_this_second = self._poisson(rate * events_per_second)
            
            for _ in range(events_this_second):
                if random.random() < 0.1:
                    event_type = random.choice(self._event_types)
                else:
                    event_type = random.choice(self._event_types[:5])
                
                if random.random() < 0.05:
                    event_type = event_type.replace('INFO', 'ERROR')
                
                if use_simulator:
                    timestamp = timestamp_simulator.generate_timestamp(current_offset)
                else:
                    timestamp = base_time + current_offset
                
                log_line = f"[{datetime.fromtimestamp(timestamp).isoformat()}] {event_type} (request_id={uuid.uuid4()})"
                
                if random.random() < 0.02:
                    log_line += " with additional context data that is unusually long " + "x" * random.randint(100, 500)
                
                yield timestamp, log_line
                event_counter += 1
                
                if event_counter % 500 == 0 and not self._burst_mode:
                    self._burst_mode = True
                    self._burst_end_offset = current_offset + 5
            
            current_offset += 1
    
    def generate_events_fast(
        self,
        duration_seconds: int = 30,
        events_per_second: int = 10
    ) -> Generator[Tuple[float, str], None, None]:
        base_time = 1700000000.0
        event_counter = 0
        
        for second_offset in range(duration_seconds):
            if self._burst_mode and second_offset > self._burst_end_offset:
                self._burst_mode = False
            
            rate = self._base_rate * (5 if self._burst_mode else 1)
            events_this_second = self._poisson(rate * events_per_second)
            
            for _ in range(events_this_second):
                if random.random() < 0.1:
                    event_type = random.choice(self._event_types)
                else:
                    event_type = random.choice(self._event_types[:5])
                
                if random.random() < 0.05:
                    event_type = event_type.replace('INFO', 'ERROR')
                
                timestamp = base_time + second_offset + (random.random() * 0.999)
                log_line = f"[{datetime.fromtimestamp(timestamp).isoformat()}] {event_type} (request_id={uuid.uuid4()})"
                
                if random.random() < 0.02:
                    log_line += " with additional context data that is unusually long " + "x" * random.randint(100, 500)
                
                yield timestamp, log_line
                event_counter += 1
                
                if event_counter % 500 == 0 and not self._burst_mode:
                    self._burst_mode = True
                    self._burst_end_offset = second_offset + 5


def format_anomaly_report(report: AnomalyReport, max_context_length: int = 300) -> str:
    context_str = json.dumps(report.context, indent=2)
    if len(context_str) > max_context_length:
        context_str = context_str[:max_context_length] + "..."
    
    return (
        f"[{report.timestamp.isoformat()}] {report.severity.value.upper()}: {report.anomaly_type}\n"
        f"  {report.description}\n"
        f"  Context: {context_str}\n"
        f"  Correlation ID: {report.correlation_id}"
    )


def demonstrate_advanced_processing():
    print("ADVANCED LOG PROCESSING DEMONSTRATION")
    print("=" * 50)
    
    processor = AdvancedLogProcessor(
        correlation_window_seconds=30,
        event_buffer_size=5000,
        processing_mode=ProcessingMode.HYBRID
    )
    simulator = DataStreamSimulator()
    timestamp_sim = TimestampSimulator(start_time=1700000000.0, speed_factor=100.0)
    
    print("\n[1] Simulating data stream with anomalies (fast mode)...")
    events_processed = 0
    anomaly_count = 0
    
    for timestamp, log_line in simulator.generate_events_fast(duration_seconds=15, events_per_second=8):
        result = processor.process_line(log_line, timestamp)
        events_processed += 1
        
        if result['is_anomaly']:
            anomaly_count += 1
            if anomaly_count <= 3:
                print(f"\n  ANOMALY DETECTED #{anomaly_count}:")
                print(f"    Line length: {result['length']}")
                print(f"    Z-score: {result['anomaly_score']:.2f}")
                print(f"    Patterns: {[p for p, m in result['patterns'].items() if m]}")
                print(f"    Correlation ID: {result['correlation_id']}")
    
    print(f"\n  Processed {events_processed} events, detected {anomaly_count} anomalies")
    
    print("\n[2] Window Statistics (last 10 seconds):")
    stats = processor.get_window_statistics(window_seconds=10)
    print(f"  Duration: {stats.duration_seconds:.2f} seconds")
    print(f"  Event counts by pattern: {stats.event_counts}")
    print(f"  Current entropy: {stats.entropy:.4f}")
    print(f"  Sample events:")
    for sample in stats.sample_events[:3]:
        print(f"    {sample[:100]}...")
    
    print("\n[3] Comprehensive Analysis Report:")
    report = processor.get_comprehensive_report()
    
    print("\n  Pattern Statistics:")
    for pattern, pattern_stats in report['pattern_statistics'].items():
        print(f"    {pattern}: {pattern_stats['total_matches']} matches, {pattern_stats['unique_matches']} unique")
    
    print("\n  Entropy State:")
    entropy_state = report['entropy_state']
    print(f"    Raw entropy: {entropy_state['current_raw_entropy']:.4f}")
    print(f"    Smoothed entropy: {entropy_state['current_smoothed_entropy']:.4f}")
    print(f"    Unique signatures: {entropy_state['unique_event_signatures']}")
    
    print("\n  Temporal Analysis:")
    print(f"    Arrival regularity: {report['temporal_analysis']['arrival_regularity']}")
    print(f"    Trend: {report['temporal_analysis']['trend']}")
    
    print("\n  Correlation Groups:")
    for pattern, count in report['correlation_groups'].items():
        print(f"    {pattern}: {count} active events")
    
    print("\n  Baseline Status:")
    baseline = report['threshold_baseline']
    print(f"    Calibrated: {baseline['calibrated']}")
    print(f"    Mean length: {baseline['mean']:.2f}")
    print(f"    Std deviation: {baseline['std']:.2f}")
    
    print(f"\n  Total Anomalies: {report['total_anomalies_detected']}")
    print(f"  Processing Mode: {report['processing_mode']}")
    
    print("\n[4] Anomaly Reports Generated:")
    anomaly_history = processor.get_anomaly_history(limit=3)
    for anomaly in anomaly_history:
        print(f"\n{format_anomaly_report(anomaly)}")
    
    print("\n" + "=" * 50)
    print("DEMONSTRATION COMPLETE")


if __name__ == "__main__":
    demonstrate_advanced_processing()
