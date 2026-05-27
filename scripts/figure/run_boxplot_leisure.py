from scripts.figure.箱型图 import plot_from_path

if __name__ == '__main__':
    path = r'c:\Users\12932\Desktop\nus\BAP\features\leisure\gpu_optimized_features_leisure_inclflagged_enhanced_features.xlsx'
    out = plot_from_path(path, apps=None, out_dir=None, max_anoms=200)
    print(out)
