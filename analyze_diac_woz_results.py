import json
import os
from pathlib import Path
from typing import List, Dict, Tuple
import statistics
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# PHQ-8 topic names
PHQ8_TOPICS = [
    "Loss of Interest",
    "Depressed Mood",
    "Sleep Problems",
    "Fatigue or Low Energy",
    "Appetite or Weight Changes",
    "Low Self-Worth",
    "Concentration Difficulties",
    "Psychomotor Changes"
]


def load_json_files(directory: str) -> List[Dict]:
    """Load all JSON files from the specified directory."""
    json_files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist")
        return json_files

    for file_path in directory_path.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_files.append(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding {file_path}: {e}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return json_files


def calculate_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Calculate precision, recall, F1 for binary or multiclass classification."""
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return {
            'accuracy': None,
            'macro_precision': None,
            'macro_recall': None,
            'macro_f1': None,
            'micro_precision': None,
            'micro_recall': None,
            'micro_f1': None
        }
    
    # Get unique labels to determine if binary or multiclass
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Macro averages (average across classes)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Micro averages (aggregate contributions of all classes)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision * 100,
        'macro_recall': macro_recall * 100,
        'macro_f1': macro_f1 * 100,
        'micro_precision': micro_precision * 100,
        'micro_recall': micro_recall * 100,
        'micro_f1': micro_f1 * 100,
        'num_samples': len(y_true),
        'num_classes': len(unique_labels)
    }


def calculate_metrics(data: List[Dict]) -> Dict:
    """Calculate overall scores, MAE, and accuracy metrics for DAIC-WOZ data."""
    if not data:
        return {}

    # PHQ-8 Total Scores
    predicted_total_scores = []
    true_total_scores = []
    total_score_errors = []
    absolute_errors = []

    # Binary classification for depression (0 = not depressed, 1 = depressed)
    y_true_binary = []
    y_pred_binary = []

    # Per-topic metrics
    per_topic_metrics = {topic: {
        'predicted_scores': [],
        'true_scores': [],
        'errors': [],
        'absolute_errors': [],
        'exact_matches': 0,
        'total': 0,
        'y_true': [],  # For multi-class classification (0, 1, 2, 3)
        'y_pred': []
    } for topic in PHQ8_TOPICS}

    # Process each evaluation result
    for item in data:
        # Total PHQ-8 scores
        if 'predictions' in item and 'phq8_total_score' in item['predictions']:
            predicted_total_scores.append(item['predictions']['phq8_total_score'])

        if 'ground_truth' in item and 'phq8_total_score' in item['ground_truth']:
            true_total_scores.append(item['ground_truth']['phq8_total_score'])

        # Binary classification (for depression detection)
        if 'ground_truth' in item and 'phq8_binary' in item['ground_truth']:
            true_binary = item['ground_truth']['phq8_binary']
            y_true_binary.append(true_binary)
            
            # Derive predicted binary from total score
            # PHQ-8 >= 10 is typically considered depression threshold
            if 'predictions' in item and 'phq8_total_score' in item['predictions']:
                pred_total = item['predictions']['phq8_total_score']
                pred_binary = 1 if pred_total >= 10 else 0
                y_pred_binary.append(pred_binary)

        # Error metrics
        if 'evaluation_metrics' in item:
            if 'total_error' in item['evaluation_metrics']:
                total_score_errors.append(item['evaluation_metrics']['total_error'])
            if 'absolute_error' in item['evaluation_metrics']:
                absolute_errors.append(item['evaluation_metrics']['absolute_error'])

            # Per-topic errors
            if 'per_topic_errors' in item['evaluation_metrics']:
                for topic_error in item['evaluation_metrics']['per_topic_errors']:
                    topic_name = topic_error['topic']
                    if topic_name in per_topic_metrics:
                        pred_score = topic_error['predicted']
                        true_score = topic_error['true']
                        
                        per_topic_metrics[topic_name]['predicted_scores'].append(pred_score)
                        per_topic_metrics[topic_name]['true_scores'].append(true_score)
                        per_topic_metrics[topic_name]['errors'].append(topic_error['error'])
                        per_topic_metrics[topic_name]['absolute_errors'].append(topic_error['absolute_error'])
                        per_topic_metrics[topic_name]['total'] += 1
                        
                        # For classification metrics (treating each score 0-3 as a class)
                        per_topic_metrics[topic_name]['y_true'].append(true_score)
                        per_topic_metrics[topic_name]['y_pred'].append(pred_score)
                        
                        if topic_error['error'] == 0:
                            per_topic_metrics[topic_name]['exact_matches'] += 1

    # Calculate binary classification metrics for depression
    binary_classification_metrics = calculate_classification_metrics(y_true_binary, y_pred_binary)

    # Calculate summary statistics for per-topic metrics
    per_topic_summary = {}
    for topic, metrics in per_topic_metrics.items():
        if metrics['total'] > 0:
            # Calculate classification metrics for this topic
            topic_classification = calculate_classification_metrics(
                metrics['y_true'], 
                metrics['y_pred']
            )
            
            per_topic_summary[topic] = {
                'accuracy': metrics['exact_matches'] / metrics['total'] * 100,
                'mae': statistics.mean(metrics['absolute_errors']) if metrics['absolute_errors'] else None,
                'mean_error': statistics.mean(metrics['errors']) if metrics['errors'] else None,
                'mean_predicted': statistics.mean(metrics['predicted_scores']) if metrics['predicted_scores'] else None,
                'mean_true': statistics.mean(metrics['true_scores']) if metrics['true_scores'] else None,
                'total_samples': metrics['total'],
                # Classification metrics
                'macro_precision': topic_classification['macro_precision'],
                'macro_recall': topic_classification['macro_recall'],
                'macro_f1': topic_classification['macro_f1'],
                'micro_precision': topic_classification['micro_precision'],
                'micro_recall': topic_classification['micro_recall'],
                'micro_f1': topic_classification['micro_f1'],
            }

    results = {
        'total_samples': len(data),
        'phq8_total_score': {
            'predicted': {
                'mean': statistics.mean(predicted_total_scores) if predicted_total_scores else None,
                'median': statistics.median(predicted_total_scores) if predicted_total_scores else None,
                'stdev': statistics.stdev(predicted_total_scores) if len(predicted_total_scores) > 1 else None,
                'min': min(predicted_total_scores) if predicted_total_scores else None,
                'max': max(predicted_total_scores) if predicted_total_scores else None,
            },
            'true': {
                'mean': statistics.mean(true_total_scores) if true_total_scores else None,
                'median': statistics.median(true_total_scores) if true_total_scores else None,
                'stdev': statistics.stdev(true_total_scores) if len(true_total_scores) > 1 else None,
                'min': min(true_total_scores) if true_total_scores else None,
                'max': max(true_total_scores) if true_total_scores else None,
            },
            'mae': statistics.mean(absolute_errors) if absolute_errors else None,
            'mean_error': statistics.mean(total_score_errors) if total_score_errors else None,
        },
        'binary_classification': binary_classification_metrics,
        'per_topic_metrics': per_topic_summary
    }

    return results


def print_results(results: Dict):
    """Print the results in a formatted way."""
    print("\n" + "="*80)
    print("DAIC-WOZ EVALUATION RESULTS ANALYSIS")
    print("="*80)
    print(f"\nTotal Samples: {results['total_samples']}")

    # PHQ-8 Total Scores
    print("\n" + "-"*80)
    print("PHQ-8 TOTAL SCORE (0-24 scale)")
    print("-"*80)

    print("\nPREDICTED SCORES:")
    pred = results['phq8_total_score']['predicted']
    print(f"  Mean:   {pred['mean']:.2f}" if pred['mean'] is not None else "  Mean:   N/A")
    print(f"  Median: {pred['median']:.2f}" if pred['median'] is not None else "  Median: N/A")
    print(f"  StdDev: {pred['stdev']:.2f}" if pred['stdev'] is not None else "  StdDev: N/A")
    print(f"  Min:    {pred['min']}" if pred['min'] is not None else "  Min:    N/A")
    print(f"  Max:    {pred['max']}" if pred['max'] is not None else "  Max:    N/A")

    print("\nTRUE SCORES:")
    true = results['phq8_total_score']['true']
    print(f"  Mean:   {true['mean']:.2f}" if true['mean'] is not None else "  Mean:   N/A")
    print(f"  Median: {true['median']:.2f}" if true['median'] is not None else "  Median: N/A")
    print(f"  StdDev: {true['stdev']:.2f}" if true['stdev'] is not None else "  StdDev: N/A")
    print(f"  Min:    {true['min']}" if true['min'] is not None else "  Min:    N/A")
    print(f"  Max:    {true['max']}" if true['max'] is not None else "  Max:    N/A")

    print("\nERROR METRICS:")
    mae = results['phq8_total_score']['mae']
    mean_error = results['phq8_total_score']['mean_error']
    print(f"  MAE (Mean Absolute Error): {mae:.2f}" if mae is not None else "  MAE: N/A")
    print(f"  Mean Error (Bias):         {mean_error:.2f}" if mean_error is not None else "  Mean Error: N/A")

    # Binary Classification for Depression
    print("\n" + "-"*80)
    print("BINARY CLASSIFICATION (Depressed vs Not Depressed)")
    print("Threshold: PHQ-8 >= 10 indicates depression")
    print("-"*80)

    bc = results['binary_classification']
    if bc['accuracy'] is not None:
        print(f"\nAccuracy:         {bc['accuracy']:.2f}%")
        print(f"Samples:          {bc['num_samples']}")
        print(f"\nMacro Metrics (average across classes):")
        print(f"  Precision:      {bc['macro_precision']:.2f}%")
        print(f"  Recall:         {bc['macro_recall']:.2f}%")
        print(f"  F1-Score:       {bc['macro_f1']:.2f}%")
        print(f"\nMicro Metrics (aggregate across all predictions):")
        print(f"  Precision:      {bc['micro_precision']:.2f}%")
        print(f"  Recall:         {bc['micro_recall']:.2f}%")
        print(f"  F1-Score:       {bc['micro_f1']:.2f}%")
    else:
        print("\nNo classification data available")

    # Per-Topic Metrics
    print("\n" + "="*80)
    print("PER-TOPIC METRICS (Each scored 0-3)")
    print("="*80)

    if results['per_topic_metrics']:
        print(f"\n{'Topic':<35} {'Acc%':<8} {'MAE':<8} {'Pred':<7} {'True':<7}")
        print("-" * 80)

        for topic in PHQ8_TOPICS:
            if topic in results['per_topic_metrics']:
                m = results['per_topic_metrics'][topic]
                print(f"{topic:<35} "
                      f"{m['accuracy']:>5.1f}%  "
                      f"{m['mae']:>6.2f}  "
                      f"{m['mean_predicted']:>6.2f} "
                      f"{m['mean_true']:>6.2f}")

        # Overall per-topic statistics
        all_accuracies = [m['accuracy'] for m in results['per_topic_metrics'].values()]
        all_maes = [m['mae'] for m in results['per_topic_metrics'].values() if m['mae'] is not None]

        print("-" * 80)
        print(f"{'AVERAGE':<35} "
              f"{statistics.mean(all_accuracies):>5.1f}%  "
              f"{statistics.mean(all_maes):>6.2f}")

        # Per-topic classification metrics
        print("\n" + "-"*80)
        print("PER-TOPIC CLASSIFICATION METRICS")
        print("-"*80)
        print(f"\n{'Topic':<35} {'Macro-F1':<12} {'Micro-F1':<12}")
        print("-" * 80)

        for topic in PHQ8_TOPICS:
            if topic in results['per_topic_metrics']:
                m = results['per_topic_metrics'][topic]
                macro_f1 = m['macro_f1'] if m['macro_f1'] is not None else 0
                micro_f1 = m['micro_f1'] if m['micro_f1'] is not None else 0
                print(f"{topic:<35} {macro_f1:>6.2f}%      {micro_f1:>6.2f}%")

        # Averages
        all_macro_f1 = [m['macro_f1'] for m in results['per_topic_metrics'].values() if m['macro_f1'] is not None]
        all_micro_f1 = [m['micro_f1'] for m in results['per_topic_metrics'].values() if m['micro_f1'] is not None]

        print("-" * 80)
        print(f"{'AVERAGE':<35} {statistics.mean(all_macro_f1):>6.2f}%      {statistics.mean(all_micro_f1):>6.2f}%")

        # Detailed per-topic metrics table
        print("\n" + "-"*80)
        print("DETAILED PER-TOPIC CLASSIFICATION METRICS")
        print("-"*80)
        print(f"\n{'Topic':<35} {'Macro-P':<10} {'Macro-R':<10} {'Macro-F1':<10}")
        print("-" * 80)

        for topic in PHQ8_TOPICS:
            if topic in results['per_topic_metrics']:
                m = results['per_topic_metrics'][topic]
                print(f"{topic:<35} "
                      f"{m['macro_precision']:>6.2f}%   "
                      f"{m['macro_recall']:>6.2f}%   "
                      f"{m['macro_f1']:>6.2f}%")

        print("\n" + "-"*80)
        print(f"\n{'Topic':<35} {'Micro-P':<10} {'Micro-R':<10} {'Micro-F1':<10}")
        print("-" * 80)

        for topic in PHQ8_TOPICS:
            if topic in results['per_topic_metrics']:
                m = results['per_topic_metrics'][topic]
                print(f"{topic:<35} "
                      f"{m['micro_precision']:>6.2f}%   "
                      f"{m['micro_recall']:>6.2f}%   "
                      f"{m['micro_f1']:>6.2f}%")

    print("\n" + "="*80 + "\n")


def main():
    """Main function to run the analysis."""
    response_dir = 'results/diac_woz'
    print(f"\nLoading JSON files from: {response_dir}")
    data = load_json_files(response_dir)

    if not data:
        print("No JSON files found or loaded successfully.")
        return

    print(f"Loaded {len(data)} JSON files successfully.")

    # Calculate metrics
    results = calculate_metrics(data)

    # Print results
    print_results(results)

    # Save results to result_analysis.json in the same directory
    output_file = "results/overall_results/daicwoz_result_analysis.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()