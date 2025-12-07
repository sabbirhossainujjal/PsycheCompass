import json
import os
from pathlib import Path
from typing import List, Dict
import statistics


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


def calculate_metrics(data: List[Dict]) -> Dict:
    """Calculate comprehensive metrics for Psych8k data."""
    if not data:
        return {}

    # Assessment metrics
    total_scores = []
    total_turns = []

    # Classification distribution
    classification_counts = {
        'None': 0,
        'Minimal': 0,
        'Mild': 0,
        'Moderate': 0,
        'Moderately Severe': 0,
        'Severe': 0
    }

    # Per-topic metrics
    per_topic_metrics = {topic: {
        'scores': [],
        'turns': [],
        'total': 0
    } for topic in PHQ8_TOPICS}

    # Therapy quality metrics
    therapy_quality_metrics = {
        'empathy': [],
        'specificity': [],
        'actionability': [],
        'appropriateness': [],
        'helpfulness': [],
        'overall_quality': []
    }

    # Similarity metrics
    similarity_metrics = {
        'semantic_similarity': [],
        'word_overlap': [],
        'jaccard_similarity': [],
        'bigram_overlap': [],
        'length_ratio': []
    }

    # Processing time metrics
    assessment_times = []
    therapy_times = []
    total_times = []

    # Process each evaluation result
    for item in data:
        # Assessment results
        if 'assessment_results' in item:
            assessment = item['assessment_results']

            # Total score
            if 'total_score' in assessment:
                total_scores.append(assessment['total_score'])

            # Total turns
            if 'total_turns' in assessment:
                total_turns.append(assessment['total_turns'])

            # Classification
            if 'classification' in assessment:
                classification = assessment['classification']
                if classification in classification_counts:
                    classification_counts[classification] += 1

            # Per-topic data
            if 'topics' in assessment:
                for topic_data in assessment['topics']:
                    topic_name = topic_data.get('topic_name')
                    if topic_name in per_topic_metrics:
                        per_topic_metrics[topic_name]['scores'].append(
                            topic_data.get('final_score', 0))
                        per_topic_metrics[topic_name]['turns'].append(
                            topic_data.get('total_turns', 0))
                        per_topic_metrics[topic_name]['total'] += 1

        # Therapy quality evaluation
        if 'therapy_results' in item and 'quality_evaluation' in item['therapy_results']:
            quality = item['therapy_results']['quality_evaluation']

            for metric in therapy_quality_metrics.keys():
                if metric in quality and quality[metric] is not None:
                    therapy_quality_metrics[metric].append(quality[metric])

        # Similarity metrics
        if 'therapy_results' in item and 'similarity_metrics' in item['therapy_results']:
            similarity = item['therapy_results']['similarity_metrics']

            for metric in similarity_metrics.keys():
                if metric in similarity and similarity[metric] is not None:
                    similarity_metrics[metric].append(similarity[metric])

        # Processing times
        if 'metadata' in item and 'duration_seconds' in item['metadata']:
            duration = item['metadata']['duration_seconds']
            if 'assessment' in duration:
                assessment_times.append(duration['assessment'])
            if 'therapy' in duration:
                therapy_times.append(duration['therapy'])
            if 'total' in duration:
                total_times.append(duration['total'])

    # Calculate summary statistics for per-topic metrics
    per_topic_summary = {}
    for topic, metrics in per_topic_metrics.items():
        if metrics['total'] > 0:
            per_topic_summary[topic] = {
                'mean_score': statistics.mean(metrics['scores']) if metrics['scores'] else None,
                'median_score': statistics.median(metrics['scores']) if metrics['scores'] else None,
                'stdev_score': statistics.stdev(metrics['scores']) if len(metrics['scores']) > 1 else None,
                'mean_turns': statistics.mean(metrics['turns']) if metrics['turns'] else None,
                'total_samples': metrics['total']
            }

    # Calculate therapy quality summary
    therapy_quality_summary = {}
    for metric, values in therapy_quality_metrics.items():
        if values:
            therapy_quality_summary[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else None,
                'min': min(values),
                'max': max(values)
            }

    # Calculate similarity summary
    similarity_summary = {}
    for metric, values in similarity_metrics.items():
        if values:
            similarity_summary[metric] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'stdev': statistics.stdev(values) if len(values) > 1 else None,
                'min': min(values),
                'max': max(values)
            }

    # Compile results
    results = {
        'total_samples': len(data),
        'assessment_metrics': {
            'total_score': {
                'mean': statistics.mean(total_scores) if total_scores else None,
                'median': statistics.median(total_scores) if total_scores else None,
                'stdev': statistics.stdev(total_scores) if len(total_scores) > 1 else None,
                'min': min(total_scores) if total_scores else None,
                'max': max(total_scores) if total_scores else None,
            },
            'total_turns': {
                'mean': statistics.mean(total_turns) if total_turns else None,
                'median': statistics.median(total_turns) if total_turns else None,
                'stdev': statistics.stdev(total_turns) if len(total_turns) > 1 else None,
                'min': min(total_turns) if total_turns else None,
                'max': max(total_turns) if total_turns else None,
            },
            'classification_distribution': classification_counts
        },
        'per_topic_metrics': per_topic_summary,
        'therapy_quality': therapy_quality_summary,
        'similarity_metrics': similarity_summary,
        'processing_times': {
            'assessment': {
                'mean': statistics.mean(assessment_times) if assessment_times else None,
                'median': statistics.median(assessment_times) if assessment_times else None,
                'min': min(assessment_times) if assessment_times else None,
                'max': max(assessment_times) if assessment_times else None,
            },
            'therapy': {
                'mean': statistics.mean(therapy_times) if therapy_times else None,
                'median': statistics.median(therapy_times) if therapy_times else None,
                'min': min(therapy_times) if therapy_times else None,
                'max': max(therapy_times) if therapy_times else None,
            },
            'total': {
                'mean': statistics.mean(total_times) if total_times else None,
                'median': statistics.median(total_times) if total_times else None,
                'min': min(total_times) if total_times else None,
                'max': max(total_times) if total_times else None,
            }
        }
    }

    return results


def print_results(results: Dict):
    """Print the results in a formatted way."""
    print("\n" + "="*80)
    print("PSYCH8K EVALUATION RESULTS ANALYSIS")
    print("="*80)
    print(f"\nTotal Samples: {results['total_samples']}")

    # Assessment Metrics
    print("\n" + "-"*80)
    print("ASSESSMENT METRICS")
    print("-"*80)

    # PHQ-8 Total Scores
    print("\nPHQ-8 TOTAL SCORE (0-24 scale):")
    score = results['assessment_metrics']['total_score']
    print(f"  Mean:   {score['mean']:.2f}" if score['mean']
          is not None else "  Mean:   N/A")
    print(f"  Median: {score['median']:.2f}" if score['median']
          is not None else "  Median: N/A")
    print(f"  StdDev: {score['stdev']:.2f}" if score['stdev']
          is not None else "  StdDev: N/A")
    print(f"  Min:    {score['min']}" if score['min']
          is not None else "  Min:    N/A")
    print(f"  Max:    {score['max']}" if score['max']
          is not None else "  Max:    N/A")

    # Total Turns
    print("\nTOTAL CONVERSATION TURNS:")
    turns = results['assessment_metrics']['total_turns']
    print(f"  Mean:   {turns['mean']:.2f}" if turns['mean']
          is not None else "  Mean:   N/A")
    print(f"  Median: {turns['median']:.2f}" if turns['median']
          is not None else "  Median: N/A")
    print(f"  StdDev: {turns['stdev']:.2f}" if turns['stdev']
          is not None else "  StdDev: N/A")
    print(f"  Min:    {turns['min']}" if turns['min']
          is not None else "  Min:    N/A")
    print(f"  Max:    {turns['max']}" if turns['max']
          is not None else "  Max:    N/A")

    # Classification Distribution
    print("\nCLASSIFICATION DISTRIBUTION:")
    dist = results['assessment_metrics']['classification_distribution']
    total_classified = sum(dist.values())
    for classification, count in dist.items():
        percentage = (count / total_classified *
                      100) if total_classified > 0 else 0
        print(f"  {classification:<20} {count:>4} ({percentage:>5.1f}%)")

    # Per-Topic Metrics
    print("\n" + "-"*80)
    print("PER-TOPIC METRICS (Each scored 0-3)")
    print("-"*80)

    if results['per_topic_metrics']:
        print(
            f"\n{'Topic':<35} {'Mean Score':<12} {'Mean Turns':<12} {'Samples':<10}")
        print("-" * 80)

        for topic in PHQ8_TOPICS:
            if topic in results['per_topic_metrics']:
                metrics = results['per_topic_metrics'][topic]
                mean_score = metrics['mean_score']
                mean_turns = metrics['mean_turns']
                total_samples = metrics['total_samples']

                print(f"{topic:<35} "
                      f"{mean_score:>6.2f}      "
                      f"{mean_turns:>6.2f}      "
                      f"{total_samples:>4}")

        # Overall per-topic statistics
        all_mean_scores = [m['mean_score'] for m in results['per_topic_metrics'].values()
                           if m['mean_score'] is not None]
        all_mean_turns = [m['mean_turns'] for m in results['per_topic_metrics'].values()
                          if m['mean_turns'] is not None]

        print("-" * 80)
        print(f"{'AVERAGE':<35} "
              f"{statistics.mean(all_mean_scores):>6.2f}      "
              f"{statistics.mean(all_mean_turns):>6.2f}")

    # Therapy Quality Metrics
    print("\n" + "-"*80)
    print("THERAPY QUALITY METRICS (Scored 1-5)")
    print("-"*80)

    if results['therapy_quality']:
        print(
            f"\n{'Metric':<25} {'Mean':<10} {'Median':<10} {'StdDev':<10} {'Min':<8} {'Max':<8}")
        print("-" * 80)

        for metric_name in ['empathy', 'specificity', 'actionability', 'appropriateness',
                            'helpfulness', 'overall_quality']:
            if metric_name in results['therapy_quality']:
                metric = results['therapy_quality'][metric_name]
                print(f"{metric_name.capitalize():<25} "
                      f"{metric['mean']:>6.2f}    "
                      f"{metric['median']:>6.2f}    "
                      f"{metric['stdev']:>6.2f}    " if metric['stdev'] is not None else f"{'-':>6}    "
                      f"{metric['min']:>6.2f}  "
                      f"{metric['max']:>6.2f}")

    # Similarity Metrics
    print("\n" + "-"*80)
    print("SIMILARITY METRICS")
    print("-"*80)

    if results['similarity_metrics']:
        print(f"\n{'Metric':<25} {'Mean':<10} {'Median':<10} {'StdDev':<10}")
        print("-" * 80)

        for metric_name, metric in results['similarity_metrics'].items():
            metric_display = metric_name.replace('_', ' ').title()
            print(f"{metric_display:<25} "
                  f"{metric['mean']:>6.4f}    "
                  f"{metric['median']:>6.4f}    "
                  f"{metric['stdev']:>6.4f}" if metric['stdev'] is not None else f"{'-':>6}")

    # Processing Times
    print("\n" + "-"*80)
    print("PROCESSING TIMES (seconds)")
    print("-"*80)

    for phase in ['assessment', 'therapy', 'total']:
        if phase in results['processing_times']:
            times = results['processing_times'][phase]
            if times['mean'] is not None:
                print(f"\n{phase.upper()}:")
                print(f"  Mean:   {times['mean']:>8.2f}s")
                print(f"  Median: {times['median']:>8.2f}s")
                print(f"  Min:    {times['min']:>8.2f}s")
                print(f"  Max:    {times['max']:>8.2f}s")

    print("\n" + "="*80 + "\n")


def main():
    """Main function to run the analysis."""
    response_dir = 'results/psych8k/individual_result'
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

    # Save results to result_analysis.json in the overall results directory
    output_file = "results/overall_results/psych8k_result_analysis.json"

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
