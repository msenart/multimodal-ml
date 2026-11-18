// template.typ
#let rapport(titre: "", auteurs: (), date: datetime.today(), body) = {
  set document(title: titre, author: auteurs)
  set text(lang: "en", size: 11pt)
  set par(justify: true)
  set heading(numbering: "1.1.")
  set math.equation(numbering: "(1)")
  set text(font: "New Computer Modern", lang: "fr", size: 11pt)
  
  // Page de garde
  align(center + horizon)[
    #text(size: 22pt, weight: "bold", titre)
    #v(1cm)
    #auteurs.join(", ")
    #v(0.5cm)
    #date.display("[day]/[month]/[year]")
  ]
  pagebreak()

  // Reste du document
  set page(numbering: "1")
  outline()
  pagebreak()

  body
}