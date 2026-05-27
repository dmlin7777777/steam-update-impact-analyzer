from 箱型图 import plot_from_path

if __name__ == '__main__':
    # matches the default hard-coded file in 箱型图.main
    fp = r"c:\Users\12932\Desktop\nus\BAP\features\leisure\gpu_optimized_features_leisure_inclflagged_enhanced_features.xlsx"
    print('Calling plot_from_path for', fp)
    out = plot_from_path(fp)
    print('Generated:', out)
