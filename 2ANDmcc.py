# McCulloch-Pitts Neural Net for ANDNOT function

def mcculloch_pitts_andnot(x1, x2):
    # Define weights
    w1 = 1
    w2 = -1
    
    # Define threshold
    threshold = 1

    # Weighted sum
    weighted_sum = (x1 * w1) + (x2 * w2)

    # Activation function
    if weighted_sum >= threshold:
        return 1
    else:
        return 0

# Testing all possible inputs
print("X1 X2 | Output (X1 AND NOT X2)")
print("-----------------------------")
for x1 in [0, 1]:
    for x2 in [0, 1]:
        output = mcculloch_pitts_andnot(x1, x2)
        print(f" {x1}  {x2} |       {output}")
