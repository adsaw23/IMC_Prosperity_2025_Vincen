#include <bits/stdc++.h>
using namespace std;

const int N = 4;

string names[N] = {"Snowballs", "Pizzas", "Silicon Nuggets", "SeaShells"};

int main() {
    double adj[N][N] = {
        {1.00, 1.45, 0.52, 0.72},    // From Snowballs
        {0.70, 1.00, 0.31, 0.48},    // From Pizzas
        {1.95, 3.10, 1.00, 1.49},    // From Silicon Nuggets
        {1.34, 1.98, 0.64, 1.00}     // From SeaShells
    };

    int start = 3;  
    vector<int> nodes;
    for (int i = 0; i < N; ++i)
        if (i != start) nodes.push_back(i);  

    double maxProduct = 0;
    vector<int> bestPath;

    for (int mask = 1; mask < (1 << nodes.size()); ++mask) {
        vector<int> subset;
        for (int i = 0; i < nodes.size(); ++i) {
            if (mask & (1 << i))
                subset.push_back(nodes[i]);
        }

        sort(subset.begin(), subset.end());
        do {
            double product = 1.0;
            int current = start;

            cout << names[start] << " -> ";
            for (int next : subset) {
                product *= adj[current][next];
                cout << names[next] << " -> ";
                current = next;
            }
            product *= adj[current][start];
            cout << names[start];
            cout << " | Product = " << product << "\n";

            if (product > maxProduct) {
                maxProduct = product;
                bestPath = {start};
                bestPath.insert(bestPath.end(), subset.begin(), subset.end());
                bestPath.push_back(start);
            }

        } while (next_permutation(subset.begin(), subset.end()));
    }

    cout << "\nBest cycle:\n";
    for (int x : bestPath)
        cout << names[x] << " -> ";
    cout << names[start] << "\n";

    cout << "Max product: " << maxProduct << endl;

    return 0;
}
