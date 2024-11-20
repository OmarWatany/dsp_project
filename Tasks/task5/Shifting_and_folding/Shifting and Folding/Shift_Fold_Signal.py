def Shift_Fold_Signal(file_name,Your_indices,Your_samples):      
    expected_indices=[]
    expected_samples=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    # return(f"Current Output Test file is: {file_name}\n")

    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        return("Shift_Fold_Signal Test case failed, your signal have different length from the expected one")

    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            return("Shift_Fold_Signal Test case failed, your signal have different indicies from the expected one")

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            return("Shift_Fold_Signal Test case failed, your signal have different values from the expected one")

    return("Shift_Fold_Signal Test case passed successfully")