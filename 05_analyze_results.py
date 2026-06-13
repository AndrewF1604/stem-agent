import json
import pandas as pd
import matplotlib.pyplot as plt

def main():
    try:
        with open("data/classified_errors.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("data/classified_errors.json not found.")
        return

    # Process data into a dataframe
    rows = []
    for item in data:
        q_id = item['id']
        q_type = item['type']
        q_level = item['level']
        
        for config, metrics_list in item['metrics'].items():
            classifications = item['errors'][config]['classifications']
            stochasticity = item['errors'][config]['stochasticity']
            
            for i, metric in enumerate(metrics_list):
                rows.append({
                    'id': q_id,
                    'type': q_type,
                    'level': q_level,
                    'config': config,
                    'iteration': i,
                    'em': metric['em'],
                    'f1': metric['f1'],
                    'judge': metric['judge'],
                    'error_type': classifications[i],
                    'stochasticity': stochasticity
                })
                
    df = pd.DataFrame(rows)
    
    # 1. Performance by config
    print("=== Performance by Configuration ===")
    perf = df.groupby('config')[['em', 'f1', 'judge']].mean()
    print(perf)
    
    # Plotting Performance
    perf.plot(kind='bar', figsize=(10,6))
    plt.title('Performance Metrics by Configuration')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig('data/performance_by_config.png')
    
    # 2. Error Distribution
    print("\n=== Error Types Distribution ===")
    errors_only = df[df['judge'] == 0]
    error_dist = errors_only['error_type'].value_counts()
    print(error_dist)
    
    plt.figure(figsize=(10,6))
    error_dist.plot(kind='pie', autopct='%1.1f%%')
    plt.title('Distribution of Error Types')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('data/error_distribution.png')
    
    # 3. Systematic vs Stochastic
    print("\n=== Systematic vs Stochastic Errors ===")
    # We only care about question-level stochasticity, so we drop duplicates for (id, config)
    q_level_df = df.drop_duplicates(subset=['id', 'config'])
    sys_stoch = q_level_df.groupby('config')['stochasticity'].value_counts().unstack().fillna(0)
    print(sys_stoch)
    
    sys_stoch.plot(kind='bar', stacked=True, figsize=(10,6))
    plt.title('Systematic vs Stochastic vs No Error by Configuration')
    plt.ylabel('Number of Questions (out of 100)')
    plt.tight_layout()
    plt.savefig('data/systematic_stochastic.png')
    
    print("\nAnalysis complete. Plots saved to data/ directory.")

if __name__ == "__main__":
    main()
