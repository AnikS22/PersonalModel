#!/usr/bin/env python3
"""
Experiment: Error-Driven Training
Test if targeting model weaknesses improves performance
"""

import sys
import time
from datetime import datetime

# Setup
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.models.model_manager import get_model_manager
from src.data.database import get_database
from src.data.template_engine import get_template_engine
from src.data.quality_filter import get_quality_filter
from src.data.data_generator import get_data_generator
from src.data.error_driven_generator import get_error_driven_generator
from src.training.lora_trainer import get_lora_trainer
from src.training.evaluator import get_evaluator


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def run_experiment():
    """Run error-driven training experiment"""
    
    print_header("Error-Driven Training Experiment")
    print("This experiment tests whether targeting model weaknesses improves performance.\n")
    
    # Initialize
    print("📋 Initializing components...")
    config = load_config()
    logger = setup_logger('experiment')
    
    # Load model
    print("🤖 Loading model...")
    model_manager = get_model_manager(config, logger)
    model_manager.load_model()
    
    # Initialize other components
    print("🔧 Setting up pipeline...")
    database = get_database(config, logger)
    template_engine = get_template_engine(config, logger)
    quality_filter = get_quality_filter(config, model_manager, logger)
    data_generator = get_data_generator(
        model_manager, template_engine, quality_filter, database, config, logger
    )
    error_driven_gen = get_error_driven_generator(
        model_manager, template_engine, quality_filter, database, config, logger
    )
    trainer = get_lora_trainer(model_manager, database, config, logger)
    evaluator = get_evaluator(model_manager, database, logger)
    
    print("✅ All components initialized\n")
    
    # PHASE 1: Baseline Evaluation
    print_header("Phase 1: Baseline Evaluation")
    
    # Create test set
    print("Creating test set...")
    test_prompts = [
        "Explain quantum computing in simple terms",
        "Write a Python function to reverse a string",
        "What is the capital of France?",
        "Describe photosynthesis",
        "How do neural networks learn?",
    ]
    
    evaluator.set_test_set([
        (prompt, "") for prompt in test_prompts
    ])
    
    print("Evaluating baseline model...")
    baseline_metrics = evaluator.evaluate(
        use_test_set=True,
        custom_prompts=test_prompts
    )
    
    print("\n📊 Baseline Metrics:")
    print(f"   Test Perplexity: {baseline_metrics.get('test_perplexity', 'N/A'):.2f}")
    if 'generation' in baseline_metrics:
        gen = baseline_metrics['generation']
        print(f"   Generation Perplexity: {gen.get('avg_perplexity', 'N/A'):.2f}")
        print(f"   Diversity (bigram): {gen.get('diversity_bigram', 0):.3f}")
    
    # Set as baseline for comparison
    evaluator.set_baseline(baseline_metrics)
    
    # PHASE 2: Identify Weaknesses
    print_header("Phase 2: Identify Model Weaknesses")
    
    print("Analyzing model performance...")
    weak_areas = error_driven_gen.identify_weak_areas(max_samples=20)
    
    if weak_areas:
        print(f"\n🔍 Found {len(weak_areas)} weak areas:")
        for i, weak in enumerate(weak_areas[:5], 1):
            print(f"\n   {i}. Perplexity: {weak['perplexity']:.1f}")
            print(f"      Input: {weak['input'][:60]}...")
            print(f"      Expected: {weak['expected'][:60]}...")
            print(f"      Actual: {weak['actual'][:60]}...")
    else:
        print("⚠️  No weak areas found (all examples have low perplexity)")
        print("   Try adding more diverse test examples or lowering threshold")
        return
    
    # PHASE 3: Generate Targeted Training Data
    print_header("Phase 3: Generate Targeted Training Data")
    
    print("Generating training examples targeting weaknesses...")
    stats = error_driven_gen.generate_and_save(num_weak_areas=5)
    
    print(f"\n📦 Generated Data:")
    print(f"   Total examples: {stats['generated_count']}")
    print(f"   Saved to database: {stats['saved_count']}")
    print(f"   Weak areas targeted: {stats['num_weak_areas_targeted']}")
    
    # PHASE 4: Train on Targeted Data
    print_header("Phase 4: Train on Error-Driven Data")
    
    if stats['saved_count'] == 0:
        print("⚠️  No training data generated, skipping training")
        return
    
    print("Training model on targeted examples...")
    print("(This may take a few minutes...)\n")
    
    try:
        # Train
        train_stats = trainer.train()
        
        print(f"\n✅ Training completed:")
        print(f"   Steps: {train_stats['num_steps']}")
        print(f"   Avg Loss: {train_stats['avg_loss']:.4f}")
        print(f"   Duration: {train_stats['duration_seconds']:.1f}s")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("   This might be due to:")
        print("   - Insufficient training data")
        print("   - Out of memory (try reducing batch size)")
        print("   - Model compatibility issues")
        return
    
    # PHASE 5: Re-evaluate
    print_header("Phase 5: Post-Training Evaluation")
    
    print("Re-evaluating model after error-driven training...")
    new_metrics = evaluator.evaluate(
        use_test_set=True,
        custom_prompts=test_prompts
    )
    
    print("\n📊 New Metrics:")
    print(f"   Test Perplexity: {new_metrics.get('test_perplexity', 'N/A'):.2f}")
    if 'generation' in new_metrics:
        gen = new_metrics['generation']
        print(f"   Generation Perplexity: {gen.get('avg_perplexity', 'N/A'):.2f}")
        print(f"   Diversity (bigram): {gen.get('diversity_bigram', 0):.3f}")
    
    # Compare to baseline
    if 'comparison' in new_metrics:
        comp = new_metrics['comparison']
        print("\n📈 Comparison to Baseline:")
        
        if 'perplexity_change_percent' in comp:
            change = comp['perplexity_change_percent']
            if comp.get('perplexity_improved', False):
                print(f"   ✅ Perplexity improved by {abs(change):.1f}%")
            else:
                print(f"   ❌ Perplexity degraded by {abs(change):.1f}%")
        
        if 'diversity_change_percent' in comp:
            change = comp['diversity_change_percent']
            if comp.get('diversity_improved', False):
                print(f"   ✅ Diversity improved by {abs(change):.1f}%")
            else:
                print(f"   ❌ Diversity degraded by {abs(change):.1f}%")
    
    # Decision
    should_accept, reason = evaluator.should_accept_model(new_metrics)
    
    print(f"\n🎯 Model Acceptance Decision:")
    if should_accept:
        print(f"   ✅ ACCEPT - {reason}")
        print("   Error-driven training improved the model!")
    else:
        print(f"   ❌ REJECT - {reason}")
        print("   Error-driven training did not improve the model")
    
    # PHASE 6: Summary
    print_header("Experiment Summary")
    
    print("🔬 Hypothesis Tested:")
    print("   Training on examples targeting model weaknesses")
    print("   improves performance on those weak areas.")
    
    print("\n📊 Results:")
    baseline_ppl = baseline_metrics.get('test_perplexity', float('inf'))
    new_ppl = new_metrics.get('test_perplexity', float('inf'))
    
    if new_ppl < baseline_ppl:
        improvement = (baseline_ppl - new_ppl) / baseline_ppl * 100
        print(f"   ✅ HYPOTHESIS SUPPORTED")
        print(f"   Perplexity improved by {improvement:.1f}%")
        print(f"   ({baseline_ppl:.2f} → {new_ppl:.2f})")
    else:
        degradation = (new_ppl - baseline_ppl) / baseline_ppl * 100
        print(f"   ❌ HYPOTHESIS NOT SUPPORTED")
        print(f"   Perplexity degraded by {degradation:.1f}%")
        print(f"   ({baseline_ppl:.2f} → {new_ppl:.2f})")
    
    print("\n💡 Next Steps:")
    if should_accept:
        print("   - Run more iterations to compound improvements")
        print("   - Try different weak area thresholds")
        print("   - Combine with other training strategies")
    else:
        print("   - Investigate why training didn't help")
        print("   - Try different quality filtering")
        print("   - Generate more training examples per weak area")
        print("   - Adjust perplexity threshold for weakness detection")
    
    print("\n" + "="*70)
    print(f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        run_experiment()
    except KeyboardInterrupt:
        print("\n\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

