#!/usr/bin/env python3
"""
Quick test to check for console errors in the model manager
"""

import sys
import os
sys.path.append('.')

def test_model_manager():
    """Test model manager functionality"""
    print("=" * 50)
    print("Testing Model Manager")
    print("=" * 50)
    
    try:
        from src.models.model_manager import ModelManager
        print("✅ ModelManager imported successfully")
        
        mm = ModelManager()
        print("✅ ModelManager initialized successfully")
        
        print(f"✅ Supported models: {list(mm.SUPPORTED_MODELS.keys())}")
        
        print("\nModel configurations:")
        for model_key, config in mm.SUPPORTED_MODELS.items():
            print(f"  {model_key}:")
            print(f"    - Name: {config['name']}")
            print(f"    - Type: {config['type']}")
            print(f"    - Task: {config['task']}")
            print(f"    - Description: {config['description']}")
        
        print("\n✅ All model configurations loaded successfully!")
        
        # Test basic functionality without downloading models
        print("\nTesting basic functionality...")
        
        # Test if model exists in supported models
        assert 'codebert' in mm.SUPPORTED_MODELS, "CodeBERT not found in supported models"
        assert 'sentence-transformer' in mm.SUPPORTED_MODELS, "Sentence transformer not found"
        
        print("✅ All basic tests passed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_manager()
    if success:
        print("\n🎉 All tests passed! No console errors detected.")
    else:
        print("\n💥 Tests failed! Check errors above.")