"""
Green Campus, Happy Campus
Handout 4: the dashboard (Streamlit)

Input : GC_PROFILES.csv (made by the Machine Learning notebook), in the same folder as this file
Run   : streamlit run app.py        (if that fails:  python -m streamlit run app.py)
Needs : pip install streamlit pandas scikit-learn matplotlib

Four pages: Overview, Fix-First Matrix, Student profiles, Check my campus.
The charts are the same pictures as in the Colab notebooks and in the student guide.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from sklearn.tree import DecisionTreeClassifier, export_text

st.set_page_config(page_title="Green Campus Dashboard", layout="wide")
st.title("Green Campus, Happy Campus")

BLUE = "#1F4E9E"
LIGHT_BLUE = "#7FA6E0"
GREY = "#B8B8B8"

# ---------------------------------------------------------------------------
# Step 2: Read the data
# ---------------------------------------------------------------------------
df = pd.read_csv("GC_PROFILES.csv")
st.caption("Based on " + str(len(df)) + " survey responses")

# ---------------------------------------------------------------------------
# Step 3: The model and the helper rules (the same as Handouts 2 and 3)
# ---------------------------------------------------------------------------
features = ["Gap", "Wellbeing"]
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(df[features], df["Profile"])

# The group with the lowest wellbeing is the one that needs the most help (blue in charts).
FOCUS = df.groupby("Profile")["Wellbeing"].mean().idxmin()


def gap_label(gap):
    if gap >= 1.5:
        return "Large gap"
    if gap >= 0.5:
        return "Some gap"
    return "Balanced"


factors = {
    "Q1": "Enough green spaces",
    "Q7": "Classroom comfort",
    "Q8": "Noise levels",
    "Q9": "Lighting",
    "Q10": "Temperature and ventilation",
    "Q11": "Places to sit and relax",
    "Q12": "Cleanliness",
    "Q23_R": "Little noise or crowding in class",
    "Q27": "Chances for green activities",
}

rows = []
for q in factors:
    rows.append([factors[q], round(df[q].corr(df["Q30"], method="spearman"), 2),
                 round(df[q].mean(), 2)])

matrix = pd.DataFrame(rows, columns=["Factor", "Importance", "Score"])
important = matrix["Importance"] >= matrix["Importance"].median()
low_score = matrix["Score"] < matrix["Score"].median()
matrix["Fix_First"] = important & low_score


def result_for(green, indoor, wellbeing):
    """green: 3 answers, indoor: dict of 4 answers, wellbeing: 6 answers (each 1 to 5)."""
    green_score = sum(green) / len(green)
    indoor_score = sum(indoor.values()) / len(indoor)
    wellbeing_score = sum(wellbeing) / len(wellbeing)
    gap = green_score - indoor_score
    profile = tree.predict(pd.DataFrame([[gap, wellbeing_score]], columns=features))[0]
    weakest = min(indoor, key=indoor.get)
    return green_score, indoor_score, wellbeing_score, gap, profile, weakest


# ---------------------------------------------------------------------------
# Charts (same pictures as the notebooks)
# ---------------------------------------------------------------------------
def gap_chart():
    fig, ax = plt.subplots(figsize=(7, 4))
    edges = np.arange(-2, 4.5, 0.5)
    counts, _, bars = ax.hist(df["Gap"], bins=edges, edgecolor="white")
    for bar, left in zip(bars, edges[:-1]):
        bar.set_facecolor(BLUE if left >= 1.5 else LIGHT_BLUE if left >= 0.5 else GREY)
    for bar, count in zip(bars, counts):
        if count > 0:
            ax.text(bar.get_x() + 0.25, count + 0.4, int(count), ha="center", fontsize=9)
    ax.axvline(0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Gap (Green score minus Indoor score).  Right of 0 = green space rated higher")
    ax.set_ylabel("Number of students")
    ax.legend(handles=[
        Patch(color=GREY, label="Balanced (" + str((df["Gap_Label"] == "Balanced").sum()) + ")"),
        Patch(color=LIGHT_BLUE, label="Some gap (" + str((df["Gap_Label"] == "Some gap").sum()) + ")"),
        Patch(color=BLUE, label="Large gap (" + str((df["Gap_Label"] == "Large gap").sum()) + ")"),
    ])
    fig.tight_layout()
    return fig


def profile_chart(extra_point=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in sorted(df["Profile"].unique()):
        part = df[df["Profile"] == name]
        ax.scatter(part["Gap"], part["Wellbeing"], alpha=0.7,
                   color=BLUE if name == FOCUS else "#9A9A9A",
                   label=name + ": " + str(len(part)))
    if extra_point is not None:
        ax.scatter([extra_point[0]], [extra_point[1]], s=220, marker="*", color="black",
                   label="You")
    ax.set_xlabel("Green-Comfort Gap  (further right = bigger gap)")
    ax.set_ylabel("Wellbeing  (higher = feels better)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    return fig


def fix_first_chart():
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for i in range(len(matrix)):
        fix = matrix["Fix_First"][i]
        ax.scatter(matrix["Score"][i], matrix["Importance"][i], s=70,
                   color=BLUE if fix else "#9A9A9A")
        shift = (5, -12) if matrix["Factor"][i] == "Cleanliness" else (5, 4)
        ax.annotate(matrix["Factor"][i], (matrix["Score"][i], matrix["Importance"][i]),
                    fontsize=8, xytext=shift, textcoords="offset points")
    ax.axvline(matrix["Score"].median(), linestyle="--", color="black", linewidth=1)
    ax.axhline(matrix["Importance"].median(), linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Score now (1 to 5)  -  further right = campus does better")
    ax.set_ylabel("Importance for happiness  -  higher = matters more")
    ax.set_title("Top-left (blue) = fix first")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Step 4: The menu
# ---------------------------------------------------------------------------
page = st.sidebar.radio("Go to", ["Overview", "Fix-First Matrix",
                                  "Student profiles", "Check my campus"])

# ---------------------------------------------------------------------------
# Step 5: The Overview page
# ---------------------------------------------------------------------------
if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students analysed", len(df))
    c2.metric("Average Green score", round(df["Green_Score"].mean(), 2))
    c3.metric("Average Indoor score", round(df["Indoor_Score"].mean(), 2))
    large = round((df["Gap_Label"] == "Large gap").mean() * 100)
    c4.metric("Students with a large gap", str(large) + "%")

    st.subheader("Green-Comfort Gap (number of students)")
    st.pyplot(gap_chart())
    st.caption("Gap = Green score - Indoor score. Large gap: 1.5 or more. "
               "Some gap: 0.5 to 1.5. Balanced: below 0.5.")

    st.subheader("Lowest-rated questions (average, 1 to 5)")
    # Q23 is a negative question, so we use the reversed Q23_R; Q29 is left out (only 20 answers).
    names = dict(factors)
    names["Q26"] = "Students care about sustainability"
    rated = [q for q in ["Q" + str(i) for i in range(1, 34)] if q not in ("Q23", "Q29")] + ["Q23_R"]
    lowest = df[rated].mean().sort_values().head(5).round(2)
    lowest.index = [names.get(q, q) for q in lowest.index]
    st.bar_chart(lowest)

# ---------------------------------------------------------------------------
# Step 6: The Fix-First Matrix page
# ---------------------------------------------------------------------------
elif page == "Fix-First Matrix":
    st.write("Which campus factors should the university improve first?")
    st.pyplot(fix_first_chart())
    st.caption("How to read it: each dot is one thing on campus. Further left = rated lower now. "
               "Higher up = matters more for happiness. Top-left (blue) = fix first.")
    st.dataframe(matrix.sort_values("Importance", ascending=False))
    fix = matrix[matrix["Fix_First"]]["Factor"].tolist()
    st.success("Fix first: " + ", ".join(fix))
    st.caption("Importance = how strongly the factor moves with happiness (Q30). "
               "Score = how well the campus does now. Important + low score = fix first. "
               "This shows two answers moving together, not that one causes the other.")

# ---------------------------------------------------------------------------
# Step 7: The Student profiles page
# ---------------------------------------------------------------------------
elif page == "Student profiles":
    columns = ["Gap", "Wellbeing", "Green_Score", "Indoor_Score", "Q30"]
    summary = df.groupby("Profile")[columns].mean().round(2)
    summary["Students"] = df["Profile"].value_counts()
    st.dataframe(summary)
    st.pyplot(profile_chart())
    st.caption("Each dot is one student. Blue = the group that needs the most help.")
    st.subheader("Decision Tree rules")
    st.code(export_text(tree, feature_names=features))

# ---------------------------------------------------------------------------
# Steps 8 and 9: The Check my campus page
# ---------------------------------------------------------------------------
elif page == "Check my campus":
    st.write("Rate each statement from 1 (strongly disagree) to 5 (strongly agree).")

    with st.form("new_student"):
        st.markdown("**Green spaces**")
        g1 = st.slider("I enjoy spending time in areas with trees, plants or gardens", 1, 5, 3)
        g2 = st.slider("Spending time in green spaces helps me feel relaxed", 1, 5, 3)
        g3 = st.slider("Natural surroundings improve my overall experience of being on campus", 1, 5, 3)

        st.markdown("**Where I study**")
        i1 = st.slider("My classroom / learning environment is physically comfortable", 1, 5, 3)
        i2 = st.slider("Noise levels on campus are generally comfortable", 1, 5, 3)
        i3 = st.slider("Lighting in my usual learning spaces is comfortable", 1, 5, 3)
        i4 = st.slider("Temperature and ventilation are comfortable where I study", 1, 5, 3)
        indoor = {"Classroom comfort": i1, "Noise levels": i2, "Lighting": i3,
                  "Temperature and ventilation": i4}

        st.markdown("**How I feel on campus**")
        w = [st.slider("The campus lets me concentrate effectively", 1, 5, 3),
             st.slider("I feel calm in pleasant areas of the campus", 1, 5, 3),
             st.slider("The campus environment helps reduce my everyday stress", 1, 5, 3),
             st.slider("I feel comfortable and psychologically safe on campus", 1, 5, 3),
             st.slider("A pleasant campus makes me feel more positive about university life", 1, 5, 3),
             st.slider("I feel a sense of belonging in shared campus spaces", 1, 5, 3)]

        submitted = st.form_submit_button("Show my result")

    if submitted:
        green_score, indoor_score, wellbeing, gap, profile, weakest = result_for(
            [g1, g2, g3], indoor, w)

        c1, c2, c3 = st.columns(3)
        c1.metric("Your Gap Index", round(gap, 2))
        c2.metric("Gap label", gap_label(gap))
        c3.metric("Closest profile", profile)

        st.subheader("Why this result?")
        st.write("Green score " + str(round(green_score, 2)) + " minus Indoor score "
                 + str(round(indoor_score, 2)) + " gives a gap of " + str(round(gap, 2)) + ".")
        st.write("The indoor factor you rated lowest is: " + weakest
                 + " (" + str(indoor[weakest]) + " out of 5).")

        if gap >= 1.5:
            st.warning("You value the green campus much more than your study spaces. "
                       "Studying in a greener or better-ventilated place may help.")
        elif gap >= 0.5:
            st.info("Your study spaces are a little behind the green campus for you.")
        else:
            st.success("Your green campus and study spaces feel balanced to you.")

        st.pyplot(profile_chart((gap, wellbeing)))
        st.caption("The black star is you, among the 114 students. "
                   "The profile is the closest group, not a diagnosis.")
