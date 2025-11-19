import re
import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For Pareto front calculation
import numpy as np

def is_pareto_efficient(costs, scores):
    """
    Return a boolean array indicating whether each point is Pareto efficient.
    """
    # Create array of shape (n_points, 2): (cost, scores)
    data = np.vstack([costs, scores]).T
    n_points = data.shape[0]
    is_efficient = np.ones(n_points, dtype=bool)
    for i, c in enumerate(data):
        if is_efficient[i]:
            # Any point that is strictly better in both dims will dominate
            # Check for any other point j: data[j] <= c in both dims, and < in at least one
            is_efficient[is_efficient] = np.any(data[is_efficient] < c, axis=1) | np.all(data[is_efficient] == c, axis=1)
            is_efficient[i] = True  # keep self
    return is_efficient

def parse_bicycle_data(js_text):
    """
    Extract the bicycleDataRaw array from the JS text.
    """
    # Find the `const bicycleDataRaw = [ ... ];` part
    m = re.search(r"const\s+bicycleDataRaw\s*=\s*(\[\s*\{.*?\}\s*\]);", js_text, re.DOTALL)
    if not m:
        raise ValueError("Could not find bicycleDataRaw in JS")
    array_text = m.group(1)
    # But JS uses unquoted keys, trailing commas, possibly single quotes — we need to massage into JSON.
    # Replace unquoted keys with quoted ones: a simple but somewhat fragile regex-based approach.
    # First, wrap keys in quotes.
    def replace_key(match):
        return f'"{match.group(1)}":'
    array_text_quoted = re.sub(r"(\b[a-zA-Z_][a-zA-Z0-9_]*)\s*:", replace_key, array_text)
    # Replace single quotes with double quotes
    # array_text_quoted = array_text_quoted.replace("'", '"')
    # Remove trailing commas before closing braces/brackets
    array_text_quoted = re.sub(r",\s*([}\]])", r"\1", array_text_quoted)
    # breakpoint()
    # Now load via json
    data = json.loads(array_text_quoted)
    return data

def download_js(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.text

def clean_cost(cost_str):
    """
    Convert cost strings like '$110' into numeric value.
    """
    # remove $ and commas, convert to float
    return float(cost_str.replace('$', '').replace(',', ''))

def main():
    url = "https://www.helmet.beam.vt.edu/js/bicycleData.js"
    js = download_js(url)
    data = parse_bicycle_data(js)
    df = pd.DataFrame(data)
    # Extract fields
    df['name'] = df['brand'] + " " + df['model']
    df['cost'] = df['cost'].apply(clean_cost)
    df['score'] = df['score'].astype(float)
    is_pareto = is_pareto_efficient(df['cost'].values, df['score'].values)
    df['pareto'] = is_pareto

    print("Pareto-optimal bikes:")
    print(df[ df['pareto'] ][['name', 'cost', 'score']])

    # Plotting
    plt.figure(figsize=(10, 6))
    # Plot all points
    plt.scatter(df['cost'], df['score'], label='All bikes', color='blue', alpha=0.6)
    # Pareto points
    pareto_df = df[df['pareto']]
    plt.scatter(pareto_df['cost'], pareto_df['score'],
                label='Pareto front', color='red', marker='D', s=100)
    # Label pareto points
    for _, row in pareto_df.iterrows():
        plt.annotate(f"{row['name']} (${row['cost']:.0f}, {row['score']:.1f})",
                     (row['cost'], row['score']),
                     textcoords="offset points", xytext=(0,5), ha='center')
    # Sort pareto points by cost for connecting lines
    pareto_sorted = pareto_df.sort_values('cost')
    plt.plot(pareto_sorted['cost'], pareto_sorted['score'], color='red', linestyle='--')

    plt.xscale("log")
    plt.xlabel('Cost ($)')
    plt.ylabel('Score (lower the better)')
    plt.title('Bicycle Cost vs Score with Pareto Front')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
