def getvec(fname):
    vec=[]
    with open(fname,'r') as f:
        lines=f.readlines()
        for line in lines:
            vec.append(line.strip())
    f.close()
    return vec