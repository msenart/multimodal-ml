#import "template.typ": *

#show: report.with(
  title: "Multimodal Machine Learning Project",
  authors: ("Author 1", "Author 2"),
)

#include "introduction.typ"
#include "metrics.typ"
#include "baseline.typ"
#include "ogm.typ"
#include "ogm_ge.typ"

#bibliography("biblio.bib")
