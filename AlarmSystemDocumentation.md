```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
```


```python
class ScoreAlarmSystem:
    def __init__(self, top_n=4, threshold_percent=30):
        self.top_n = top_n
        self.threshold_percent = threshold_percent
        self.scores = []
        self.best_scores = []

    def add_score(self, lesson_number, score):
        self.scores.append({'lesson_number': lesson_number, 'score': score, 'timestamp': datetime.now()})
        self._update_best_scores(score)
        return self._generate_alarm(score)

    def _update_best_scores(self, score):
        self.best_scores.append(score)
        self.best_scores = sorted(self.best_scores, reverse=True)[:self.top_n]

    def _generate_alarm(self, score):
        if not self.best_scores:
            return None
            
        if len(self.scores) < 4:  # less than 4 lessons --> absolute deviation
            base_score = self.scores[0]['score']
            deviation = ((score - base_score) / base_score) * 100
            category = self._classify_absolute_change(deviation)
        else:  # for 4+ --> mediana of 4best
            median_best = np.median(self.best_scores)
            deviation = ((score - median_best) / median_best) * 100
            category = self._classify_std_based_change(score, median_best, np.std(self.best_scores))

        is_alarm = abs(deviation) > self.threshold_percent

        return {
            'current_score': score,
            'deviation_percent': deviation,
            'category': category,
            'is_alarm': is_alarm,
            'message': f"Score deviated by {deviation:.1f}% ({category})."
        }

    def _classify_absolute_change(self, deviation):
        if deviation < -75: #percentage
            return "plunge"
        elif -75 <= deviation < -35:
            return "decline"
        elif -35 <= deviation <= 30:
            return "stable"
        elif 30 < deviation <= 75:
            return "rise"
        else:
            return "surge"

    def _classify_std_based_change(self, score, median, std_dev): 
        if score < median - 2 * std_dev:
            return "plunge"
        elif median - 2 * std_dev <= score <= median - std_dev:
            return "decline"
        elif median - std_dev <= score <= median + std_dev:
            return "stable"
        elif median + std_dev <= score <= median + 2 * std_dev:
            return "rise"
        else:
            return "surge"

    def get_stats(self):
        scores_df = pd.DataFrame(self.scores)
        median_best = np.median(self.best_scores) if self.best_scores else None
        mean_best = np.mean(self.best_scores) if self.best_scores else None
        return {
            'total_scores': len(self.scores),
            'best_scores': self.best_scores,
            'median_best': median_best,
            'mean_best': mean_best,
            'overall_stats': scores_df.describe() if not scores_df.empty else None
        }

def analyze_and_visualize_scores(data_path):
    alarm_system = ScoreAlarmSystem(top_n=4, threshold_percent=30)
    test_data = pd.read_csv(data_path)

    for _, row in test_data.iterrows():
        alarm_system.add_score(row['Lesson_number'], row['tnsw'])

    stats = alarm_system.get_stats()
    total_lessons = stats['total_scores']
    scores = test_data['tnsw']
    lesson_numbers = test_data['Lesson_number']

    if total_lessons >= 4:
        median_best4 = stats['median_best']
        tnsw_std = np.std(alarm_system.best_scores)
        reference_line = median_best4
        reference_label = f'Median Best 4 ({median_best4:.1f})'
        
        categories = []
        for score in scores:
            if score < median_best4 - 2 * tnsw_std:
                categories.append("plunge")
            elif median_best4 - 2 * tnsw_std <= score <= median_best4 - tnsw_std:
                categories.append("decline")
            elif median_best4 - tnsw_std <= score <= median_best4 + tnsw_std:
                categories.append("stable")
            elif median_best4 + tnsw_std <= score <= median_best4 + 2 * tnsw_std:
                categories.append("rise")
            else:
                categories.append("surge")
    else:
        base_score = scores.iloc[0]
        reference_line = base_score
        reference_label = f'Base Score ({base_score:.1f})'
        
        categories = []
        for score in scores:
            deviation = ((score - base_score) / base_score) * 100
            if deviation < -75:
                categories.append("plunge")
            elif -75 <= deviation < -35:
                categories.append("decline")
            elif -35 <= deviation <= 30:
                categories.append("stable")
            elif 30 < deviation <= 75:
                categories.append("rise")
            else:
                categories.append("surge")

    category_colors = {
        "plunge": "#ef4444",   # Red
        "decline": "#f97316",  # Orange
        "stable": "#22c55e",   # Green
        "rise": "#3b82f6",     # Blue
        "surge": "#8b5cf6"     # Purple
    }
    bar_colors = [category_colors[cat] for cat in categories]

    plt.figure(figsize=(12, 6))
    plt.bar(lesson_numbers, scores, label='Total Words', color=bar_colors, alpha=0.8)
    plt.axhline(reference_line, color='red', linestyle='--', label=reference_label)

    deviation_percentages = ((scores - reference_line) / reference_line) * 100
    for lesson, score, deviation in zip(lesson_numbers, scores, deviation_percentages):
        if abs(deviation) > 30:
            plt.text(lesson, score + 50, f"{deviation:.1f}%", ha='center', va='bottom', fontsize=8)
    used_categories = set(categories)
    for category in used_categories:
        plt.bar(0, 0, color=category_colors[category], label=category.capitalize())

    method_text = "Based on Best 4 Scores" if total_lessons >= 4 else "Based on Absolute Deviation"
    plt.title(f'Progress Analysis ({method_text})', fontsize=14)
    plt.xlabel('Lesson Number', fontsize=12)
    plt.ylabel('Total Words (TNSW)', fontsize=12)
    plt.xticks(lesson_numbers, rotation=45)
    plt.legend(title="Categories", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    return plt
```


```python
if __name__ == "__main__":
    data_path = '/Users/Downloads/total_unique_exported_1 (5).csv'
    plt = analyze_and_visualize_scores(data_path)
    plt.show()
```


    
![png](output_2_0.png)
    

