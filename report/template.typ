// template.typ
#let report(title: "", authors: (), date: datetime.today(), body) = {
  set document(title: title, author: authors)
  set text(font: "New Computer Modern", lang: "en", size: 11pt)
  set par(justify: true)
  set heading(numbering: "1.1.")
  set math.equation(numbering: "(1)")

  // Title page
  align(center + horizon)[
    #text(size: 22pt, weight: "bold", title)
    #v(1cm)
    #authors.join(", ")
    #v(0.5cm)
    #date.display("[month repr:long] [day padding:none], [year]")
  ]
  pagebreak()

  // Rest of the document
  set page(numbering: "1")
  outline()
  pagebreak()

  body
}
