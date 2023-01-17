# Write the header
with open("prediction.csv", "a") as pred:
    pred.write("id,A,B,C,D\n")
    # We have 17965143 rows of test data, starting from 0.
    # Write 0.25 for each category for each row
    for i in range(17965143):
        pred.write(str(i) + ",0.25,0.25,0.25,0.25\n")
# This gives a score of 1.38629