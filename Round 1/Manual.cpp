#include <bits/stdc++.h>
using namespace std;

const int N = 4;
const int start = 3; 
const int maxTrades = 5; 

string names[N] = {"Snowballs", "Pizzas", "Silicon Nuggets", "SeaShells"};

double adj[N][N] = {
    {1.00, 1.45, 0.52, 0.72},    // From Snowballs
    {0.70, 1.00, 0.31, 0.48},    // From Pizzas
    {1.95, 3.10, 1.00, 1.49},    // From Silicon Nuggets
    {1.34, 1.98, 0.64, 1.00}     // From SeaShells
};

double maxProduct = 0;
vector<int> bestPath;

void dfs(int current, vector<int>& path, double product, int trades) {
    if (trades > maxTrades) return;

    if (trades == maxTrades && current == start) {
        for (int i = 0; i < path.size(); ++i) {
            cout << names[path[i]] << " -> ";
        }
        cout << names[start] << " | Product = " << product << "\n";

        if (product > maxProduct) {
            maxProduct = product;
            bestPath = path;
            bestPath.push_back(start);
        }
        return;
    }

    for (int next = 0; next < N; ++next) {
        path.push_back(next);
        dfs(next, path, product * adj[current][next], trades + 1);
        path.pop_back();
    }
}

int main() {
    vector<int> path = {start};
    dfs(start, path, 1.0, 0);

    cout << "\nBest cycle:\n";
    for (int x : bestPath)
        cout << names[x] << " -> ";
    cout << names[start] << "\n";

    cout << "Max product: " << maxProduct << endl;
    return 0;
}
