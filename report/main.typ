#import "template.typ": *

#show: rapport.with(
  titre: "Multimodal Machine Learning Project",
  auteurs: ("Toi", "Ton binôme"),
)

#outline()

#include "introduction.typ"
#include "metrics.typ"
#include "baseline.typ"
#include "ogm_ge.typ"

#bibliography("biblio.bib")