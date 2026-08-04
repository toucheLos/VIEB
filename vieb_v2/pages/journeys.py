"""Journeys page -- per-animal trajectories through the discovered states."""

from pages import PlaceholderPage


class JourneysPage(PlaceholderPage):
    TITLE = "Journeys"
    SUMMARY = ("Follow individual animals through the state space over a "
               "session and across days, to see how a behavioural repertoire "
               "changes with experience.")
    BLOCKED_ON = ("Needs a labelled run plus per-animal metadata.\n"
                  "Run the pipeline from Analysis first.")
