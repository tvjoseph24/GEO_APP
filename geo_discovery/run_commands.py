'''

GEO_STORE=geo_projects_test streamlit run geo_app.py

# start
git switch main
git pull origin main
git switch -c theme-wpp
git branch --show-current          # confirm

# work — commit often, small commits
git add -A
git status                          # read it
git commit -m "Add theme.py with WPP colour tokens and CSS"

# ... more work, more commits ...
git commit -m "Pill tabs and de-emoji section headers"

# push (backup + a place to point others at)
git push -u origin theme-wpp

#### Test then merge ########

git switch main
git pull origin main
git merge --no-ff theme-wpp -m "Merge WPP theme: colour tokens, pill tabs, icon removal"
git push origin main

git tag -a v1.2-wpp-theme -m "WPP Production theme applied"
git push origin v1.2-wpp-theme

'''