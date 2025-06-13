#!/usr/bin/env python3
"""
Test script to verify plotly fixes and real-time demo feature
"""

import plotly.graph_objects as go
import numpy as np

def test_plotly_fix():
    """Test that plotly title_font works correctly"""
    print("Testing plotly title_font fix...")
    
    try:
        # Create a simple plot with the correct title_font syntax
        fig = go.Figure(data=[
            go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3])
        ])
        
        fig.update_layout(
            xaxis=dict(
                title="X Axis",
                title_font=dict(color='#999', size=14),  # This should work now
                tickfont=dict(color='#999')
            ),
            yaxis=dict(
                title="Y Axis", 
                title_font=dict(color='#999', size=14),  # This should work now
                tickfont=dict(color='#999')
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Try to render to verify no errors
        fig.to_json()
        print("✅ Plotly title_font fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Plotly test failed: {str(e)}")
        return False

def test_real_time_demo_import():
    """Test that the real-time demo module can be imported"""
    print("\nTesting real-time demo module import...")
    
    try:
        # Test import
        from app.real_time_demo_analysis import RealTimeAnalysisDemo, run_real_time_demo
        
        # Test instantiation
        demo = RealTimeAnalysisDemo()
        
        print("✅ Real-time demo module imported successfully!")
        print(f"   - RealTimeAnalysisDemo class available")
        print(f"   - run_real_time_demo function available")
        print(f"   - Demo patch size: {demo.patch_size}")
        print(f"   - Demo scales: {demo.scales}")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Other error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Running CRC Platform Tests")
    print("=" * 50)
    
    results = []
    
    # Test 1: Plotly fix
    results.append(test_plotly_fix())
    
    # Test 2: Real-time demo module
    results.append(test_real_time_demo_import())
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Total tests: {len(results)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")
    
    if all(results):
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 