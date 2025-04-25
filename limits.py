def load_external_limits(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            values = line.strip().split()
            x.append(float(values[0]))
            y.append(float(values[1]))
    
    return x, y