"""States & Motifs page.

Will host the video player showing states, motifs and stories together.
(Formerly "State Characterization" in v1.)
"""

from pages import PlaceholderPage


class StatesMotifsPage(PlaceholderPage):
    TITLE = "States & Motifs"
    SUMMARY = ("Browse discovered states with their exemplar clips, the motifs "
               "they form, and the stories those motifs tell -- in one view "
               "rather than three.")
    BLOCKED_ON = ("Needs clip extraction and a labelled run.\n"
                  "Run the pipeline from Analysis, or:\n"
                  "python -m vieb_v2.cli run --pose <dir>")
