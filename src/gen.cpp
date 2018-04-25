#include "bits/stdc++.h"

#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int, int>
#define vi vector<int>
#define all(a) (a).begin(), (a).end()
#define F first
#define S second
#define sz(x) (int)x.size()
#define hell 1000000007
#define endl '\n'
#define rep(i, a, b) for (int i = a; i < b; i++)
using namespace std;

void solve()
{
	srand(time(NULL));
	int users, items, ratings, queries;
	cin >> users >> items >> ratings >> queries;
	cout << users << " " << items << " " << ratings << endl;
	vector<pii> x;
	rep(i, 1, users + 1)
	{
		rep(j, 1, items + 1)
		{
			x.pb(mp(i, j));
		}
	}
	random_shuffle(all(x));
	x.resize(ratings);
	sort(all(x));
	rep(i, 0, ratings)
	{
		cout << x[i].F << " " << x[i].S << endl;
	}
	sort(all(x), [](pii a, pii b) {
		if (a.S == b.S)
			return a.F < b.F;
		return a.S < b.S;
	});
	rep(i, 0, ratings)
	{
		cout << x[i].F << " " << x[i].S << endl;
	}
	random_shuffle(all(x));
	rep(i, 0, queries)
	{
		cout << x[i].F << " " << x[i].S << endl;
	}
	cout << -1 << " " << -1 << endl;
}

int main()
{
	ios_base::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);
	int t = 1;
	//	cin>>t;
	while (t--)
	{
		solve();
	}
	return 0;
}
