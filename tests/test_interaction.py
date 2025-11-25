import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from model import interact

def test_interaction():
    print("Testing Deffuant-Weisbuch Interaction Logic...")
    
    # Test Case 1: Opinions close enough (should converge)
    op1 = 0.2
    op2 = 0.4
    bound = 0.5
    mu = 0.5
    
    print(f"\nCase 1: Opinions {op1}, {op2} | Bound {bound} | Mu {mu}")
    new1, new2 = interact(op1, op2, bound, mu)
    print(f"Result: {new1:.2f}, {new2:.2f}")
    
    # Expected: 0.2 + 0.5*(0.4-0.2) = 0.3
    #           0.4 + 0.5*(0.2-0.4) = 0.3
    if abs(new1 - 0.3) < 1e-9 and abs(new2 - 0.3) < 1e-9:
        print("PASS: Opinions converged as expected.")
    else:
        print("FAIL: Opinions did not converge correctly.")

    # Test Case 2: Opinions too far (should not change)
    op3 = 0.1
    op4 = 0.9
    bound = 0.5
    mu = 0.5
    
    print(f"\nCase 2: Opinions {op3}, {op4} | Bound {bound} | Mu {mu}")
    new3, new4 = interact(op3, op4, bound, mu)
    print(f"Result: {new3:.2f}, {new4:.2f}")
    
    if new3 == op3 and new4 == op4:
        print("PASS: Opinions remained unchanged.")
    else:
        print("FAIL: Opinions changed unexpectedly.")

if __name__ == "__main__":
    test_interaction()
