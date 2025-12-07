import json
import os
from pathlib import Path
from typing import List, Dict
import statistics
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import math


def make_json_serializable(obj):
    """Recursively convert numpy types and arrays to native Python types for JSON serialization."""
    # numpy is imported as np
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    # numpy scalar types
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    # numpy arrays -> lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # other numpy types with .item()
    try:
        if hasattr(obj, 'item') and not isinstance(obj, (str, bytes, bytearray)):
            return make_json_serializable(obj.item())
    except Exception:
        pass
    return obj


def _simple_sentence_bleu(list_of_references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing=True):
    """A lightweight sentence-level BLEU approximation to use when NLTK is unavailable.

    Args:
        list_of_references: list of reference token lists (e.g., [["the", "cat"], ...])
        hypothesis: list of hypothesis tokens
        weights: tuple of n-gram weights (sums to 1)
        smoothing: whether to apply add-one smoothing for n-gram counts

    Returns:
        BLEU score (0..1)
    """
    max_n = len(weights)

    # Build reference n-gram counts (take max counts across references)
    refs_ngrams = [{} for _ in range(max_n)]
    for ref in list_of_references:
        for n in range(1, max_n + 1):
            counts = {}
            for i in range(len(ref) - n + 1):
                ng = tuple(ref[i:i + n])
                counts[ng] = counts.get(ng, 0) + 1
            # merge taking max
            for ng, c in counts.items():
                refs_ngrams[n - 1][ng] = max(refs_ngrams[n - 1].get(ng, 0), c)

    precisions = []
    for n in range(1, max_n + 1):
        hyp_counts = {}
        for i in range(len(hypothesis) - n + 1):
            ng = tuple(hypothesis[i:i + n])
            hyp_counts[ng] = hyp_counts.get(ng, 0) + 1

        # Count clipped matches
        match = 0
        total = 0
        for ng, cnt in hyp_counts.items():
            total += cnt
            match += min(cnt, refs_ngrams[n - 1].get(ng, 0))

        if total == 0:
            precisions.append(0.0)
        else:
            if smoothing:
                precisions.append((match + 1) / (total + 1))
            else:
                precisions.append(match / total)

    # If any precision is zero (and not smoothed), BLEU is zero; avoid log(0)
    log_prec_sum = 0.0
    for p, w in zip(precisions, weights):
        if p == 0:
            # If smoothing disabled and p == 0, return 0
            if not smoothing:
                return 0.0
            # else use tiny value
            p = 1e-9
        log_prec_sum += w * math.log(p)

    geo_mean = math.exp(log_prec_sum)

    # Brevity penalty
    ref_lens = [len(r) for r in list_of_references]
    ref_len = min(ref_lens, key=lambda rl: (
        abs(rl - len(hypothesis)), rl)) if ref_lens else 0
    hyp_len = len(hypothesis)
    if hyp_len == 0:
        bp = 0.0
    elif hyp_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - (ref_len / hyp_len))

    return bp * geo_mean


def _simple_rouge_1_fmeasure(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-1 F1 (token-level) between reference and hypothesis using simple token overlap."""
    ref_tokens = str(reference).split()
    hyp_tokens = str(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    # Count tokens
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    match = 0
    for t in hyp_tokens:
        if ref_counts.get(t, 0) > 0:
            match += 1
            ref_counts[t] -= 1

    precision = match / len(hyp_tokens) if hyp_tokens else 0.0
    recall = match / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    """Compute length of longest common subsequence between two token lists."""
    # Dynamic programming LCS (space-optimized somewhat)
    if not a or not b:
        return 0
    # Use 2-row DP
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[-1]


def _simple_rouge_l_fmeasure(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1 based on LCS between reference and hypothesis tokens."""
    ref_tokens = str(reference).split()
    hyp_tokens = str(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0

    lcs = _lcs_length(ref_tokens, hyp_tokens)
    precision = lcs / len(hyp_tokens) if hyp_tokens else 0.0
    recall = lcs / len(ref_tokens) if ref_tokens else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


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
    """Calculate overall scores, MAE, and semantic similarity for all metrics."""
    if not data:
        return {}

    # Extract scores
    llm_empathy_scores = []
    llm_specificity_scores = []
    llm_overall_scores = []
    gt_empathy_scores = []
    gt_specificity_scores = []
    gt_overall_scores = []
    empathy_diffs = []
    specificity_diffs = []
    overall_diffs = []

    # Extract responses for semantic similarity
    generated_responses = []
    reference_responses = []

    for item in data:
        # LLM scores
        if 'llm_empathy_score' in item:
            llm_empathy_scores.append(item['llm_empathy_score'])
        if 'llm_specificity_score' in item:
            llm_specificity_scores.append(item['llm_specificity_score'])
        if 'llm_overall_score' in item:
            llm_overall_scores.append(item['llm_overall_score'])

        # Ground truth scores
        if 'gt_empathy_score' in item:
            gt_empathy_scores.append(item['gt_empathy_score'])
        if 'gt_specificity_score' in item:
            gt_specificity_scores.append(item['gt_specificity_score'])
        if 'gt_overall_score' in item:
            gt_overall_scores.append(item['gt_overall_score'])

        # Differences (for MAE calculation)
        if 'empathy_diff' in item:
            empathy_diffs.append(abs(item['empathy_diff']))
        if 'specificity_diff' in item:
            specificity_diffs.append(abs(item['specificity_diff']))
        if 'overall_diff' in item:
            overall_diffs.append(abs(item['overall_diff']))

        # Responses for semantic similarity
        if 'generated_response' in item and 'reference_response' in item:
            generated_responses.append(item['generated_response'])
            reference_responses.append(item['reference_response'])

    # Calculate semantic similarity
    print("\nCalculating semantic similarity...")
    semantic_similarities = []
    if generated_responses and reference_responses:
        try:
            # Load sentence transformer model
            model = SentenceTransformer('all-MiniLM-L6-v2')

            # Encode responses
            gen_embeddings = model.encode(generated_responses)
            ref_embeddings = model.encode(reference_responses)

            # Calculate cosine similarity for each pair
            for gen_emb, ref_emb in zip(gen_embeddings, ref_embeddings):
                similarity = cosine_similarity([gen_emb], [ref_emb])[0][0]
                semantic_similarities.append(similarity)

            print(
                f"Calculated semantic similarity for {len(semantic_similarities)} response pairs")
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")

        # Calculate BLEU and ROUGE scores
        print("\nCalculating BLEU and ROUGE scores...")
        bleu_scores = []
        rouge1_f1_scores = []
        rougel_f1_scores = []

        if generated_responses and reference_responses:
            try:
                # BLEU (using nltk) and ROUGE (using rouge_score)
                try:
                    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                except Exception:
                    sentence_bleu = None

                try:
                    from rouge_score import rouge_scorer
                    rouge_available = True
                except Exception:
                    rouge_scorer = None
                    rouge_available = False

                smooth = None
                if sentence_bleu is not None:
                    from nltk.translate.bleu_score import SmoothingFunction
                    smooth = SmoothingFunction().method1

                for gen, ref in zip(generated_responses, reference_responses):
                    # Basic tokenization (whitespace)
                    ref_tokens = str(ref).split()
                    gen_tokens = str(gen).split()

                    # BLEU
                    if ref_tokens and gen_tokens:
                        try:
                            # Use up to 4-gram BLEU, smoothing for short sentences
                            if sentence_bleu is not None:
                                score = sentence_bleu([ref_tokens], gen_tokens, weights=(
                                    0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
                            else:
                                # Fallback to simple implementation
                                score = _simple_sentence_bleu([ref_tokens], gen_tokens, weights=(
                                    0.25, 0.25, 0.25, 0.25), smoothing=True)
                            bleu_scores.append(score)
                        except Exception:
                            bleu_scores.append(0.0)

                    # ROUGE
                    if rouge_available:
                        try:
                            scorer = rouge_scorer.RougeScorer(
                                ['rouge1', 'rougeL'], use_stemmer=True)
                            sc = scorer.score(str(ref), str(gen))
                            rouge1_f1_scores.append(sc['rouge1'].fmeasure)
                            rougel_f1_scores.append(sc['rougeL'].fmeasure)
                        except Exception:
                            rouge1_f1_scores.append(0.0)
                            rougel_f1_scores.append(0.0)
                    else:
                        # Fallback simple ROUGE implementations
                        try:
                            rouge1_f1_scores.append(
                                _simple_rouge_1_fmeasure(ref, gen))
                            rougel_f1_scores.append(
                                _simple_rouge_l_fmeasure(ref, gen))
                        except Exception:
                            rouge1_f1_scores.append(0.0)
                            rougel_f1_scores.append(0.0)

                print(
                    f"Calculated BLEU for {len(bleu_scores)} pairs and ROUGE for {len(rouge1_f1_scores)} pairs")
            except Exception as e:
                print(f"Error calculating BLEU/ROUGE: {e}")

    results = {
        'total_samples': len(data),
        'llm_scores': {
            'empathy': {
                'mean': statistics.mean(llm_empathy_scores) if llm_empathy_scores else None,
                'median': statistics.median(llm_empathy_scores) if llm_empathy_scores else None,
                'stdev': statistics.stdev(llm_empathy_scores) if len(llm_empathy_scores) > 1 else None,
                'min': min(llm_empathy_scores) if llm_empathy_scores else None,
                'max': max(llm_empathy_scores) if llm_empathy_scores else None,
            },
            'specificity': {
                'mean': statistics.mean(llm_specificity_scores) if llm_specificity_scores else None,
                'median': statistics.median(llm_specificity_scores) if llm_specificity_scores else None,
                'stdev': statistics.stdev(llm_specificity_scores) if len(llm_specificity_scores) > 1 else None,
                'min': min(llm_specificity_scores) if llm_specificity_scores else None,
                'max': max(llm_specificity_scores) if llm_specificity_scores else None,
            },
            'overall': {
                'mean': statistics.mean(llm_overall_scores) if llm_overall_scores else None,
                'median': statistics.median(llm_overall_scores) if llm_overall_scores else None,
                'stdev': statistics.stdev(llm_overall_scores) if len(llm_overall_scores) > 1 else None,
                'min': min(llm_overall_scores) if llm_overall_scores else None,
                'max': max(llm_overall_scores) if llm_overall_scores else None,
            }
        },
        'ground_truth_scores': {
            'empathy': {
                'mean': statistics.mean(gt_empathy_scores) if gt_empathy_scores else None,
                'median': statistics.median(gt_empathy_scores) if gt_empathy_scores else None,
                'stdev': statistics.stdev(gt_empathy_scores) if len(gt_empathy_scores) > 1 else None,
                'min': min(gt_empathy_scores) if gt_empathy_scores else None,
                'max': max(gt_empathy_scores) if gt_empathy_scores else None,
            },
            'specificity': {
                'mean': statistics.mean(gt_specificity_scores) if gt_specificity_scores else None,
                'median': statistics.median(gt_specificity_scores) if gt_specificity_scores else None,
                'stdev': statistics.stdev(gt_specificity_scores) if len(gt_specificity_scores) > 1 else None,
                'min': min(gt_specificity_scores) if gt_specificity_scores else None,
                'max': max(gt_specificity_scores) if gt_specificity_scores else None,
            },
            'overall': {
                'mean': statistics.mean(gt_overall_scores) if gt_overall_scores else None,
                'median': statistics.median(gt_overall_scores) if gt_overall_scores else None,
                'stdev': statistics.stdev(gt_overall_scores) if len(gt_overall_scores) > 1 else None,
                'min': min(gt_overall_scores) if gt_overall_scores else None,
                'max': max(gt_overall_scores) if gt_overall_scores else None,
            }
        },
        'mae': {
            'empathy': statistics.mean(empathy_diffs) if empathy_diffs else None,
            'specificity': statistics.mean(specificity_diffs) if specificity_diffs else None,
            'overall': statistics.mean(overall_diffs) if overall_diffs else None,
        },
        'semantic_similarity': {
            'mean': statistics.mean(semantic_similarities) if semantic_similarities else None,
            'median': statistics.median(semantic_similarities) if semantic_similarities else None,
            'stdev': statistics.stdev(semantic_similarities) if len(semantic_similarities) > 1 else None,
            'min': min(semantic_similarities) if semantic_similarities else None,
            'max': max(semantic_similarities) if semantic_similarities else None,
            'count': len(semantic_similarities)
        },
        'bleu': {
            'mean': statistics.mean(bleu_scores) if bleu_scores else None,
            'median': statistics.median(bleu_scores) if bleu_scores else None,
            'stdev': statistics.stdev(bleu_scores) if len(bleu_scores) > 1 else None,
            'min': min(bleu_scores) if bleu_scores else None,
            'max': max(bleu_scores) if bleu_scores else None,
            'count': len(bleu_scores)
        },
        'rouge': {
            'rouge1_f1': {
                'mean': statistics.mean(rouge1_f1_scores) if rouge1_f1_scores else None,
                'median': statistics.median(rouge1_f1_scores) if rouge1_f1_scores else None,
                'stdev': statistics.stdev(rouge1_f1_scores) if len(rouge1_f1_scores) > 1 else None,
                'min': min(rouge1_f1_scores) if rouge1_f1_scores else None,
                'max': max(rouge1_f1_scores) if rouge1_f1_scores else None,
                'count': len(rouge1_f1_scores)
            },
            'rougeL_f1': {
                'mean': statistics.mean(rougel_f1_scores) if rougel_f1_scores else None,
                'median': statistics.median(rougel_f1_scores) if rougel_f1_scores else None,
                'stdev': statistics.stdev(rougel_f1_scores) if len(rougel_f1_scores) > 1 else None,
                'min': min(rougel_f1_scores) if rougel_f1_scores else None,
                'max': max(rougel_f1_scores) if rougel_f1_scores else None,
                'count': len(rougel_f1_scores)
            }
        }
    }

    return results


def print_results(results: Dict):
    """Print the results in a formatted way."""
    print("\n" + "="*70)
    print("COUNSELBENCH RESULTS ANALYSIS")
    print("="*70)
    print(f"\nTotal Samples: {results['total_samples']}")

    print("\n" + "-"*70)
    print("LLM SCORES")
    print("-"*70)

    for metric in ['empathy', 'specificity', 'overall']:
        scores = results['llm_scores'][metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {scores['mean']:.4f}" if scores['mean']
              is not None else "  Mean:   N/A")
        print(f"  Median: {scores['median']:.4f}" if scores['median']
              is not None else "  Median: N/A")
        print(f"  StdDev: {scores['stdev']:.4f}" if scores['stdev']
              is not None else "  StdDev: N/A")
        print(f"  Min:    {scores['min']:.4f}" if scores['min']
              is not None else "  Min:    N/A")
        print(f"  Max:    {scores['max']:.4f}" if scores['max']
              is not None else "  Max:    N/A")

    print("\n" + "-"*70)
    print("GROUND TRUTH SCORES")
    print("-"*70)

    for metric in ['empathy', 'specificity', 'overall']:
        scores = results['ground_truth_scores'][metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean:   {scores['mean']:.4f}" if scores['mean']
              is not None else "  Mean:   N/A")
        print(f"  Median: {scores['median']:.4f}" if scores['median']
              is not None else "  Median: N/A")
        print(f"  StdDev: {scores['stdev']:.4f}" if scores['stdev']
              is not None else "  StdDev: N/A")
        print(f"  Min:    {scores['min']:.4f}" if scores['min']
              is not None else "  Min:    N/A")
        print(f"  Max:    {scores['max']:.4f}" if scores['max']
              is not None else "  Max:    N/A")

    print("\n" + "-"*70)
    print("MEAN ABSOLUTE ERROR (MAE)")
    print("-"*70)
    print(f"\nEmpathy MAE:     {results['mae']['empathy']:.4f}" if results['mae']
          ['empathy'] is not None else "\nEmpathy MAE:     N/A")
    print(f"Specificity MAE: {results['mae']['specificity']:.4f}" if results['mae']
          ['specificity'] is not None else "Specificity MAE: N/A")
    print(f"Overall MAE:     {results['mae']['overall']:.4f}" if results['mae']
          ['overall'] is not None else "Overall MAE:     N/A")

    print("\n" + "-"*70)
    print("SEMANTIC SIMILARITY (Generated vs Reference)")
    print("-"*70)
    sem_sim = results.get('semantic_similarity', {})
    print(f"\nMean:   {sem_sim.get('mean'):.4f}" if sem_sim.get(
        'mean') is not None else "\nMean:   N/A")
    print(f"Median: {sem_sim.get('median'):.4f}" if sem_sim.get(
        'median') is not None else "Median: N/A")
    print(f"StdDev: {sem_sim.get('stdev'):.4f}" if sem_sim.get(
        'stdev') is not None else "StdDev: N/A")
    print(f"Min:    {sem_sim.get('min'):.4f}" if sem_sim.get(
        'min') is not None else "Min:    N/A")
    print(f"Max:    {sem_sim.get('max'):.4f}" if sem_sim.get(
        'max') is not None else "Max:    N/A")
    print(f"Count:  {sem_sim.get('count', 0)}")

    # BLEU and ROUGE summaries
    print("\n" + "-"*70)
    print("BLEU & ROUGE")
    print("-"*70)

    bleu = results.get('bleu', {})
    print(f"\nBLEU (corpus mean): {bleu.get('mean'):.4f}" if bleu.get(
        'mean') is not None else "\nBLEU (corpus mean): N/A")
    print(f"BLEU Median:         {bleu.get('median'):.4f}" if bleu.get(
        'median') is not None else "BLEU Median: N/A")

    rouge = results.get('rouge', {})
    r1 = rouge.get('rouge1_f1', {})
    rl = rouge.get('rougeL_f1', {})

    print(f"\nROUGE-1 F1 Mean: {r1.get('mean'):.4f}" if r1.get('mean')
          is not None else "\nROUGE-1 F1 Mean: N/A")
    print(f"ROUGE-L F1 Mean: {rl.get('mean'):.4f}" if rl.get('mean')
          is not None else "ROUGE-L F1 Mean: N/A")

    print("\n" + "="*70 + "\n")


def main():
    """Main function to run the analysis."""

    response_dir = 'results/counselbench/individual_result'
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
    output_file = "results/overall_results/counselbench_result_analysis.json"
    # Ensure output directory exists
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Convert any numpy types to native Python types for JSON serialization
    serializable_results = make_json_serializable(results)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
