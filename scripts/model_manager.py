#!/usr/bin/env python3
"""
Model Management CLI Tool

Commands:
    list       - List models with filters
    best       - Find best model by performance
    compare    - Compare multiple models
    cleanup    - Remove old models
    tag        - Add tags to a model
    show       - Show detailed model information
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

import argparse
from pilotscope.ModelRegistry import ModelRegistry


def cmd_list(args):
    """List models"""
    registry = ModelRegistry()
    
    models = registry.list_models(
        algorithm=args.algo,
        dataset=args.dataset,
        tags=args.tags,
        sort_by=args.sort
    )
    
    if not models:
        print("📭 No models found")
        return
    
    print(f"\n{'='*100}")
    print(f"Found {len(models)} models")
    print(f"{'='*100}")
    print(f"{'Model ID':<32} {'Algorithm':<10} {'Train Dataset':<15} {'Tests':<6} {'Tags'}")
    print(f"{'-'*100}")
    
    for model in models:
        model_id = model['model_id']
        algo = model['algorithm']
        training = model.get('training', {})
        train_ds = training.get('dataset', 'N/A') if training else 'N/A'
        num_tests = len(model.get('testing', []))
        tags = ', '.join(model.get('tags', [])) or '-'
        
        print(f"{model_id:<32} {algo:<10} {train_ds:<15} {num_tests:<6} {tags}")
    
    print(f"{'='*100}\n")


def cmd_best(args):
    """Find best model"""
    registry = ModelRegistry()
    
    best = registry.get_best_model(
        algorithm=args.algo,
        test_dataset=args.dataset,
        metric=args.metric
    )
    
    if not best:
        print(f"❌ No models found for {args.algo}")
        if args.dataset:
            print(f"   with test results on {args.dataset}")
        return
    
    print(f"\n{'='*80}")
    print(f"🏆 Best Model: {best['model_id']}")
    print(f"{'='*80}")
    
    # Training info
    if best.get('training'):
        train = best['training']
        print(f"\nTraining:")
        print(f"  Dataset:      {train.get('dataset')}")
        print(f"  Hyperparams:  {train.get('hyperparams')}")
        if train.get('training_time'):
            print(f"  Training Time: {train['training_time']:.1f}s")
    
    # Test results
    if best.get('testing'):
        print(f"\nTest Results:")
        for test in best['testing']:
            if not args.dataset or test['dataset'] == args.dataset:
                perf = test['performance']
                print(f"  {test['dataset']}:")
                print(f"    Total Time:   {perf.get('total_time', 0):.2f}s")
                print(f"    Average Time: {perf.get('average_time', 0):.4f}s")
                print(f"    Queries:      {test['num_queries']}")
    
    # Tags
    if best.get('tags'):
        print(f"\nTags: {', '.join(best['tags'])}")
    
    # Notes
    if best.get('notes'):
        print(f"\nNotes: {best['notes']}")
    
    print(f"{'='*80}\n")


def cmd_compare(args):
    """Compare multiple models"""
    registry = ModelRegistry()
    registry.compare_models(args.models, test_dataset=args.dataset)


def cmd_cleanup(args):
    """Clean up old models"""
    registry = ModelRegistry()
    
    print(f"🗑️  Cleaning up {args.algo} models...")
    print(f"   Keeping top {args.keep} models by {args.metric}")
    
    if not args.yes:
        response = input("\nProceed? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled")
            return
    
    deleted = registry.cleanup_old_models(
        algorithm=args.algo,
        keep_top_n=args.keep,
        by_metric=args.metric
    )
    
    print(f"\n✅ Deleted {len(deleted)} models")


def cmd_tag(args):
    """Add tags to model"""
    registry = ModelRegistry()
    
    if args.remove:
        registry.remove_tag(args.model_id, *args.tags)
    else:
        registry.tag_model(args.model_id, *args.tags)


def cmd_show(args):
    """Show detailed model information"""
    registry = ModelRegistry()
    
    model = registry.get_model(args.model_id)
    
    if not model:
        print(f"❌ Model not found: {args.model_id}")
        return
    
    print(f"\n{'='*80}")
    print(f"Model: {model['model_id']}")
    print(f"{'='*80}")
    
    print(f"\nAlgorithm: {model['algorithm']}")
    print(f"Created: {model.get('created_at', 'N/A')}")
    
    # Training
    if model.get('training'):
        train = model['training']
        print(f"\nTraining:")
        print(f"  Dataset:      {train.get('dataset')}")
        print(f"  Trained At:   {train.get('trained_at', 'N/A')}")
        print(f"  Num Queries:  {train.get('num_queries', 'N/A')}")
        print(f"  Training Time: {train.get('training_time', 0):.1f}s" if train.get('training_time') else "  Training Time: N/A")
        print(f"  Hyperparameters:")
        for key, value in train.get('hyperparams', {}).items():
            print(f"    {key}: {value}")
    
    # Testing
    if model.get('testing'):
        print(f"\nTest Results ({len(model['testing'])} tests):")
        for i, test in enumerate(model['testing'], 1):
            print(f"\n  Test #{i} - {test['dataset']}:")
            print(f"    Tested At:    {test.get('tested_at', 'N/A')}")
            print(f"    Num Queries:  {test['num_queries']}")
            perf = test['performance']
            print(f"    Total Time:   {perf.get('total_time', 0):.2f}s")
            print(f"    Average Time: {perf.get('average_time', 0):.4f}s")
    
    # Tags
    if model.get('tags'):
        print(f"\nTags: {', '.join(model['tags'])}")
    
    # Notes
    if model.get('notes'):
        print(f"\nNotes:\n{model['notes']}")
    
    # File paths
    print(f"\nFiles:")
    print(f"  Model:    {model.get('model_path')}")
    print(f"  Metadata: {model.get('model_path')}.json")
    
    print(f"{'='*80}\n")


def cmd_summary(args):
    """Show registry summary"""
    registry = ModelRegistry()
    registry.print_summary(algorithm=args.algo)


def main():
    parser = argparse.ArgumentParser(
        description='Model Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all MSCN models
  python model_manager.py list --algo mscn
  
  # Find best model
  python model_manager.py best --algo mscn --dataset production
  
  # Compare models
  python model_manager.py compare mscn_20241019_103000 mscn_20241019_110000
  
  # Clean up old models
  python model_manager.py cleanup --algo mscn --keep 5
  
  # Add tags
  python model_manager.py tag mscn_20241019_103000 production best
  
  # Show model details
  python model_manager.py show mscn_20241019_103000
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # list command
    list_parser = subparsers.add_parser('list', help='List models')
    list_parser.add_argument('--algo', help='Filter by algorithm')
    list_parser.add_argument('--dataset', help='Filter by training dataset')
    list_parser.add_argument('--tags', nargs='+', help='Filter by tags')
    list_parser.add_argument('--sort', default='trained_at', 
                            choices=['trained_at', 'performance'],
                            help='Sort by (default: trained_at)')
    list_parser.set_defaults(func=cmd_list)
    
    # best command
    best_parser = subparsers.add_parser('best', help='Find best model')
    best_parser.add_argument('--algo', required=True, help='Algorithm')
    best_parser.add_argument('--dataset', help='Test dataset filter')
    best_parser.add_argument('--metric', default='total_time',
                            help='Metric to optimize (default: total_time)')
    best_parser.set_defaults(func=cmd_best)
    
    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('models', nargs='+', help='Model IDs to compare')
    compare_parser.add_argument('--dataset', help='Filter by test dataset')
    compare_parser.set_defaults(func=cmd_compare)
    
    # cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old models')
    cleanup_parser.add_argument('--algo', required=True, help='Algorithm')
    cleanup_parser.add_argument('--keep', type=int, default=5, 
                               help='Keep top N models (default: 5)')
    cleanup_parser.add_argument('--metric', default='total_time',
                               help='Ranking metric (default: total_time)')
    cleanup_parser.add_argument('--yes', action='store_true',
                               help='Skip confirmation')
    cleanup_parser.set_defaults(func=cmd_cleanup)
    
    # tag command
    tag_parser = subparsers.add_parser('tag', help='Add/remove tags')
    tag_parser.add_argument('model_id', help='Model ID')
    tag_parser.add_argument('tags', nargs='+', help='Tags to add/remove')
    tag_parser.add_argument('--remove', action='store_true', help='Remove tags')
    tag_parser.set_defaults(func=cmd_tag)
    
    # show command
    show_parser = subparsers.add_parser('show', help='Show model details')
    show_parser.add_argument('model_id', help='Model ID')
    show_parser.set_defaults(func=cmd_show)
    
    # summary command
    summary_parser = subparsers.add_parser('summary', help='Show registry summary')
    summary_parser.add_argument('--algo', help='Filter by algorithm')
    summary_parser.set_defaults(func=cmd_summary)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()

