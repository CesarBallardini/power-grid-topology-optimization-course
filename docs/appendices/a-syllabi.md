# Appendix A — University Syllabi

This appendix records the official course pages, syllabi and catalog entries consulted to choose the
textbooks in the [study plan](../index.md). For each course it gives the **scope**, **prerequisites**,
**stated goals** and **bibliography**, taken from the course page itself. The survey was carried out in
September 2026 and covers well over 100 course pages from more than 50 universities in the United States, Europe (with emphasis on Belgium and
Germany, where Elia and 50Hertz operate, and on Spain) and Latin America (Argentina, Chile, Brazil, Mexico).

Access status of the books (archive.org) and Spanish translations are consolidated in
[Appendix B](b-books.md).

## How to read this appendix

- **Book tags:** R = required ("básica", "obligatoria"); Rec = recommended ("complementaria", "consulta");
  "(list)" = the syllabus does not separate required from recommended.
- **n/s** or **n.s.** = not stated on the page.
- **Counts** in the rankings are distinct universities citing a book; a university counts once even if several
  of its courses cite it. The three UTN regional faculties count as one university.
- **Dates:** several pages are old (some Latin American programs date from 2008–2015). The date or plan year is
  given where it matters.
- **Language:** "ES" and "PT" mark Spanish and Portuguese editions cited by the syllabus.

## Summary: most-cited books by subject

| Subject | Most-cited books (number of universities) | Study plan chapters |
|---|---|---|
| Linear algebra | Lay (6); Strang (6); Anton (3); Hoffman & Kunze (3) | 2 |
| Numerical methods | Burden & Faires (4); Chapra & Canale (2) | 4, 5 |
| Graph theory / networks | West (2); Diestel (2); Easley & Kleinberg (2) | 3 |
| Complex variables | Brown & Churchill (4); Kreyszig (2) | 1 |
| Electric circuits | Dorf & Svoboda (6); Nilsson & Riedel (5); Alexander & Sadiku (5); Hayt, Kemmerly & Durbin (4); Irwin (4) | 8–10 |
| Electromagnetism / Physics II | Serway & Jewett (2); Tipler & Mosca (2); Young & Freedman (2) | 7 |
| Electric machines and transformers | Fitzgerald, Kingsley & Umans (6); Chapman (5) | 11 |
| Power system analysis | Grainger & Stevenson (12); Glover, Sarma & Overbye (12); Gómez-Expósito, Conejo & Cañizares (8); Kundur (8, stability); Elgerd (6); Bergen & Vittal (5) | 13–15 |
| Operation, security, OPF | Wood, Wollenberg & Sheblé (11); Gómez-Expósito et al. (8) | 14–16, 19 |
| Electricity markets | Kirschen & Strbac (6); Stoft (3) | 12 |
| Linear and integer optimization | Winston (7); Hillier & Lieberman (6); Bertsimas & Tsitsiklis (4); Bazaraa, Jarvis & Sherali (4) | 21 |
| Convex and nonlinear optimization | Boyd & Vandenberghe (6); Luenberger & Ye (3) | 21 |
| Evolutionary computation | Eiben & Smith (5); Goldberg (3) | 22 |
| Scientific computing, HPC, GPU | Kirk & Hwu (2); most HPC courses use no textbook | 27 |

Observations that shaped the plan:

- **No syllabus covers PTDF/LODF with a dedicated text** beyond Wood & Wollenberg's security chapter, and
  graduate computational-methods courses (TAMU ECEN 615, UIUC ECE 530, GT ECE 6320) rely on lecture notes.
  Chapters 16–17 are therefore built on Overbye's free slides and on papers.
- **No syllabus covers quality-diversity or MAP-Elites**, and none teaches JAX. Chapters 23 and 27 are built
  on papers and official documentation.
- The closest match to ToOp's search among the surveyed courses is **USP PEA3422** (optimization methods
  applied to power systems: MILP, heuristic network reconfiguration, genetic algorithms, multi-objective).
- Latin American programs cite Spanish translations of the same US textbooks (Grainger & Stevenson, Glover,
  Hayt, Nilsson, Chapman, Fitzgerald, Hillier & Lieberman, Winston, Taha), plus Spanish originals
  (Gómez-Expósito; Fraile Mora; Barrero).

## Coverage gaps

Some pages could not be read: they required JavaScript logins or returned errors, or no syllabus was found
within the survey's search budget.

- **Not reached or unreadable:** DTU (math and circuits courses; login redirect), Comillas guía docente
  (JavaScript app), Politecnico di Milano, RWTH Aachen, TU Berlin, KIT (circuits module has no literature
  list), TU Dortmund, UGent, Imperial College, Manchester, UPM (power systems), UW-Madison ECE 427, CMU power
  courses, FIUBA plan 2020/2023 programs and the Física II / Análisis Matemático III plans, UNAM undergraduate
  power systems PDFs (HTTP 404), KU Leuven H01W4B (HTTP 500), Texas A&M ECEN 214 (PDF text not extractable).
- **Official page unreachable, secondary source used:** TU Delft courses are listed from the student
  association's (ETV) book-sales pages, which are not official. A Stanford CEE 272R syllabus seen only as a
  third-party copy was not counted.
- **Courses without named books:** Purdue ECE 20001, UIUC PHYS 212 and CS 450, ETH power system analysis and
  several DTU, KTH and ETH graduate courses use lecture notes only.

---

## Part 1 — Mathematics, physics, circuits and machines

### 1. Linear algebra

#### Syllabi consulted

1. **MIT 18.06 Linear Algebra, Spring 2010** — https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/pages/syllabus/
   - Books: R: Strang, *Introduction to Linear Algebra* 4e (2009). Rec: 5e (2016). Also Strang, *Differential Equations and Linear Algebra* (2014).
   - Scope: elimination, vector spaces and bases, least squares, orthogonalization, determinants, eigenvalues, symmetric matrices, linear transformations. Undergraduate; 3 lectures + 1 recitation per week.
   - Prereq: 18.02.
   - Goals: computational and conceptual matrix skills (elimination, the four subspaces, least squares, diagonalization), applied to engineering problems including graphs and networks.

2. **Georgia Tech MATH 1554, Fall 2026** — https://syllabus.gatech.edu/sites/default/files/2026-04/MATH%201554%20Syllabus%20for%20Fall%202026_2.pdf (GT-Europe version: https://syllabus.gatech.edu/sites/default/files/2026-04/Math1554_summer2026_Syllabus-HJ_7.pdf)
   - Books: Lay, *Linear Algebra and Its Applications* 6e; the syllabus says "textbook is optional". The GT-Europe version adds the free Margalit & Rabinoff, *Interactive Linear Algebra*.
   - Scope: row reduction, LU and SVD, linear transformations, determinants, eigenvalues, orthogonal projections and least squares. 4 credits, first year.
   - Prereq: Math SAT 620 / ACT 26, or MATH 1113 or MATH 1552 (stated in the Europe version).
   - Goals: construct, evaluate and analyze linear-algebra expressions, write proofs, and model real-world problems.

3. **Purdue MA 26500, Fall 2024** — https://www.math.purdue.edu/academic/courses/semester/202510/ma26500/index.html
   - Books: R: Lay 6e with MyLab.
   - Scope: linear systems, matrices, vector spaces, determinants, eigenvalues, diagonalization. 3 credits.
   - Prereq: n/s (the course is closed to students with credit in MA 26200/27200/35000/35100).
   - Goals: n/s.

4. **Texas A&M MATH 304, Spring 2024 (section 510)** — https://people.tamu.edu/~grigorch//Math%20304-510_2024.html
   - Books: R: Leon & de Pillis, *Linear Algebra with Applications* 10e (2020).
   - Scope: systems, matrices, determinants, vector spaces, linear maps, orthogonality, eigenvalues. 3 credits.
   - Prereq: MATH 152.
   - Goals: solve systems, work with vector spaces, orthogonalization and eigenvalues, including applications to ODEs.

5. **UC Berkeley EECS 16A (site shows Spring 2026)** — https://eecs16a.org/
   - Books: main reference Boyd & Vandenberghe, *VMLS* (free PDF). Supplements: Strang, *ILA*; Lipschutz & Lipson, *Schaum's Linear Algebra* 5e; Lee & Varaiya, *Structure and Interpretation of Signals and Systems*; *Data-Driven Science and Engineering*.
   - Scope: vectors and signals, inner products, complex exponentials, discrete-time Fourier series, least squares, eigen-analysis, SVD/PCA, LTI systems. The page title reads "Foundations of Signals, Dynamical Systems, and Information Processing".
   - Prereq: n/s. Goals: n/s.

6. **KU Leuven H01A4B Toegepaste algebra, 2026-27** — https://onderwijsaanbod.kuleuven.be/syllabi/n/H01A4BN.htm
   - Books: R: Lay, Lay & McDonald 6e Global edition, plus a Dutch supplement.
   - Scope: linear equations through SVD and quadratic forms. 6 ECTS, first semester of the engineering bachelor.
   - Prereq: secondary-school maths (at least 6 hours per week).
   - Goals: least squares, factorizations, translating engineering problems into algebra.

7. **ETH 401-0151-00L Lineare Algebra (EE&IT bachelor), HS 2025** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=192867&semkez=2025W&ansicht=KATALOGDATEN&lang=de
   - Books (list): Gradinaru lecture script; Nipp & Stoffer, *Lineare Algebra* 5e (2002); Olver & Shakiban, *Applied Linear Algebra* 2e (2018).
   - Scope: Gauss elimination, LU/QR, least squares, eigenvalues, SVD, numerical aspects. 5 ECTS.
   - Prereq: basic scientific Python.
   - Goals: solve and geometrically interpret linear systems and matrix operations.
   - Note: ETH's computer-science linear algebra course (401-0131) uses Strang, *ILA* 6e.

8. **TU Delft EE1M11 Linear Algebra & Analysis A** — https://etv.tudelft.nl/en-US/education/default/course?id=503 (ETV book list, not official)
   - Books: Poole, *Linear Algebra: A Modern Introduction* (plus Stewart, *Calculus*).
   - Scope, prereq, goals: n/s.

9. **KTH SF1624 Algebra och geometri (archived 2010-11 page)** — https://www.kth.se/social/course/SF1624/page/kurslitteratur/
   - Books: Anton & Rorres, *Elementary Linear Algebra with Applications* 10e.
   - Scope: systems through quadratic forms. 7.5 hp.
   - Prereq, goals: n/s.

10. **UTN FRBA Álgebra y Geometría Analítica, 2023 (common to all engineering degrees)** — https://www.frba.utn.edu.ar/wp-content/uploads/2023/10/Programa-Algebra-y-Geometria-Analitica.pdf
    - R:
      - Anton, *Introducción al Álgebra Lineal* (Limusa 2011)
      - Grossman, *Álgebra Lineal con Aplicaciones* (2012)
      - Kolman & Hill 8a
      - Kozak, Pastorelli & Vardanega (2007)
      - Lay, Lay & McDonald, *Álgebra lineal y sus aplicaciones* 5a (2016)
      - Poole 4a (2017)
      - Rojo, *Álgebra II*
    - Rec: Grossman & Flores 8a, Hoffman & Kunze (1973), Lipschutz, Strang *Álgebra lineal y sus aplicaciones* 4a, plus about 12 others.
    - Scope: matrices, linear systems, vectors, lines and planes, quadratic forms, vector spaces, linear maps, eigenvalues. First year, 120 clock hours.
    - Prereq: n/s.
    - Goals: abstraction through vector spaces; apply linear models and analytic geometry; use software.

11. **FIUBA 81.02 Álgebra II, 1C/2020** — https://cms.fi.uba.ar/uploads/8102_abc54c1706.pdf (same list in 61.08: https://cms.fi.uba.ar/uploads/6108_78cec112e1.pdf)
    - Books (Rec only): Jeronimo, Sabia & Tesauri (FCEyN 2008); Hoffman & Kunze (1984); Maltsev (MIR); Lay (Spanish ed., 1999); Strang *Álgebra lineal y sus aplicaciones* (1989); Grossman 3a (1990).
    - Scope: vector spaces, inner products, least squares, eigenvalues, Hermitian matrices, SVD, linear ODE systems. 8 hours per week.
    - Prereq: n/s.
    - Goals: linear algebra needed in engineering; combine theory and computation.

12. **U. de Chile MA1102 Álgebra Lineal (in force since 2006, revised 2009)** — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=3462
    - Books: Dartnell, Goles & San Martín course notes (2005); Brinkmann & Klotz (1971); Hoffman & Kunze (1973); Nering (1963).
    - Scope: matrices, LU, vector spaces, determinants, eigenvalues, quadratic forms, conics. 6 SCT credits.
    - Prereq: MA1101, MA1001.
    - Goals: model linear phenomena, diagonal and Jordan forms, eigenvalues.

13. **USP (Escola Politécnica) MAT3457 / MAT3458 Álgebra Linear I/II** — https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAT3457 and https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAT3458
    - Books (list): Camargo & Boulos; Nicholson; Anton & Rorres 10a (Portuguese); Strang *Álgebra Linear e suas Aplicações* 4a (Portuguese); Lay 2a (LTC, Portuguese); Callioli et al.; Barone Jr.
    - Scope: MAT3457 covers V3 vectors, analytic geometry, row reduction, matrices, vector spaces. MAT3458 covers inner products, linear maps, diagonalization, symmetric operators, ODE systems. 4 credits / 60 h each.
    - Prereq: not shown on the page.
    - Goals: row reduction and 3D geometry (MAT3457); inner products, eigenvalues and engineering relevance (MAT3458).

#### Book ranking

| # | Book | Cites | Universities (edition / language) |
|---|---|---|---|
| 1 | Lay, *Linear Algebra and Its Applications* | 6 | GT (6e), Purdue (6e R), KU Leuven (6e R), UTN (Spanish 5a R), FIUBA (Spanish 1999), USP (Portuguese 2a) |
| 1 | Strang, *Introduction to Linear Algebra* / *Linear Algebra and Its Applications* (two different books) | 6 | MIT (ILA 4e R), Berkeley (ILA), ETH (ILA 6e, computer-science dept.), UTN (LAA Spanish 4a), FIUBA (LAA Spanish), USP (LAA Portuguese 4a) |
| 3 | Anton (& Rorres), *Elementary Linear Algebra* | 3 | KTH (10e), UTN (Spanish, Limusa), USP (Portuguese 10a) |
| 3 | Hoffman & Kunze, *Linear Algebra* | 3 | UTN, FIUBA, UChile (Spanish) |
| 5 | Grossman, *Álgebra lineal* (Spanish only) | 2 | UTN, FIUBA |
| 5 | Poole, *Linear Algebra: A Modern Introduction* | 2 | TU Delft, UTN (Spanish 4a) |
| 5 | Lipschutz, *Schaum's* | 2 | Berkeley, UTN |

Cited once each: Leon; Boyd & Vandenberghe; Kolman & Hill; Kozak et al.; Olver & Shakiban; Nipp & Stoffer; Callioli; Nicholson.

---

### 2. Numerical methods / numerical linear algebra

#### Syllabi consulted

1. **MIT 18.335J Intro to Numerical Methods, Spring 2019** — https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/syllabus/
   - Books: R: Trefethen & Bau, *Numerical Linear Algebra* (SIAM 1997). Additional: Barrett et al., *Templates for the Solution of Linear Systems*; Bai et al., *Templates for Eigenproblems*.
   - Scope: direct and iterative solvers, eigenvalues, QR/SVD, stability, floating point, sparse matrices. Graduate.
   - Prereq: 18.06 or 18.700.
   - Goals: numerical linear algebra with attention to stability, accuracy and efficiency (uses Julia).

2. **MIT 18.085 Computational Science & Engineering I, Fall 2008** — https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/pages/syllabus/
   - Books: R: Strang, *Computational Science and Engineering* (2007).
   - Scope: applied linear algebra, differential equations, Fourier methods, algorithms.
   - Prereq: 18.02, 18.03.
   - Goals: mathematics through applications and efficient computation.

3. **UC Berkeley Math 128A, Summer 2011** — https://math.berkeley.edu/~wilken/128A.Su11
   - Books: R: Burden & Faires 9e. Rec: Dorfman, MATLAB programming.
   - Scope: round-off error, nonlinear equations, interpolation, quadrature, linear systems, ODEs.
   - Prereq: Math 53 & 54.
   - Goals: not formally stated.

4. **KTH SF1547 Numeriska metoder, grundkurs** — https://www.kth.se/social/course/SF1547/page/kurslitteratur-201/ and https://www.kth.se/student/kurser/kurs/SF1547
   - Books: Sauer, *Numerical Analysis* 2e; Edsberg et al. exercise collection (PDF).
   - Scope: linear and nonlinear equations, interpolation, ODEs, optimization. 6 hp.
   - Prereq: SF1625 plus a programming course; SF1624 recommended.
   - Goals: identify subproblems, choose and implement methods, assess reliability.

5. **KU Leuven H01D8B Numerieke wiskunde, 2026-27** — https://onderwijsaanbod.kuleuven.be/syllabi/n/H01D8BN.htm
   - Books: course text (Acco); no named textbook.
   - Scope: nonlinear equations, errors, linear systems, interpolation, quadrature, ODEs, eigenvalues, PDEs. 4 ECTS.
   - Prereq: analysis, applied algebra, programming (MATLAB).
   - Goals: errors, stability, telling ill-conditioning apart from algorithmic instability.

6. **ETH 401-0654-00L Numerische Methoden (EE&IT), FS 2026** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=198888&semkez=2026S&ansicht=KATALOGDATEN&lang=de
   - Books: "reading list announced in lecture" (none named).
   - Scope: quadrature, Newton's method, ODE solvers (explicit, implicit, stability, step control). 4 ECTS.
   - Prereq: scientific Python; analysis; linear algebra (Gauss, LU).
   - Goals: know, implement and apply basic engineering numerics, focused on ODEs.

7. **FIUBA 75.12 Análisis Numérico I, 1C/2020** — https://cms.fi.uba.ar/uploads/7512_7f9c246c95.pdf
   - Main list: Burden & Faires, *Análisis Numérico* 9a (Cengage 2012); Higham, *Accuracy and Stability of Numerical Algorithms*; Saad, *Iterative Methods for Sparse Linear Systems* 2e; Samarski (Mir); Zill, ODEs.
   - Rec: Dennis & Moré on quasi-Newton methods, Goldberg on floating point, Trefethen articles, and others.
   - Scope: errors, linear systems, roots, approximation, quadrature, ODEs.
   - Prereq: n/s.
   - Goals: develop and apply numerical techniques for engineering.

8. **UPC 820009 Cálculo Numérico. Ecuaciones Diferenciales (EEBE; required in the Electrical Engineering degree), 2026** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/820009
   - R: Vázquez et al. (2009); Huerta, Sarrate & Rodríguez-Ferran, *Métodos numéricos* (Edicions UPC, free online); Arias et al. (UPC, free online); Zill & Cullen 7a.
   - Rec: Burden & Faires *Análisis numérico* 7a; Chapra & Canale 6a.
   - Scope: computer numerics, ODEs, integral transforms, PDEs. 6 ECTS.
   - Prereq: n/s.
   - Goals: program and use basic methods with judgment; solve ODEs and PDEs analytically and numerically.

9. **USP MAP3121 Métodos Numéricos e Aplicações** — https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAP3121
   - Books: Burden & Faires (Portuguese 7th, 2008); Chapra & Canale, *Métodos Numéricos para Engenharia* (the page says "12th ed., 2009").
   - Scope: Gaussian elimination, Gauss-Seidel, conjugate gradient, Newton's method, splines and least squares, quadrature, Runge-Kutta. 4 credits / 60 h.
   - Prereq: not shown.
   - Goals: introduce basic numerical methods and applications.

#### Book ranking

| # | Book | Cites | Universities |
|---|---|---|---|
| 1 | Burden & Faires, *Numerical Analysis* | 4 | Berkeley (9e R), FIUBA (Spanish 9a), UPC (Spanish 7a Rec), USP (Portuguese 7th) |
| 2 | Chapra & Canale, *Numerical Methods for Engineers* | 2 | UPC (Spanish 6a), USP (Portuguese) |
| — | Trefethen & Bau | 1 | MIT (R) |
| — | Sauer | 1 | KTH |
| — | Saad, *Iterative Methods for Sparse Linear Systems* | 1 | FIUBA |
| — | Higham, *Accuracy and Stability* | 1 | FIUBA |
| — | Strang, *CSE* | 1 | MIT |
| — | Barrett et al., *Templates* | 1 | MIT |

Only MIT 18.335 and FIUBA 75.12 explicitly cover sparse matrices. Newton's method appears in FIUBA, ETH and USP.

---

### 3. Graph theory / network science

#### Syllabi consulted

1. **Stanford CS224W, Fall 2025** — https://web.stanford.edu/class/cs224w/
   - Books (optional, all free online): Hamilton, *Graph Representation Learning*; Easley & Kleinberg, *Networks, Crowds, and Markets*; Barabási, *Network Science*.
   - Scope: node embeddings, graph neural networks, knowledge graphs, graph transformers. Graduate.
   - Prereq: CS107/145, probability, linear algebra.
   - Goals: machine-learning and algorithmic analysis of large networks.

2. **MIT 14.15J/6.207J Networks, Spring 2018** — https://ocw.mit.edu/courses/14-15j-networks-spring-2018/pages/syllabus/
   - Books: R: Newman, *Networks: An Introduction* (2010). Rec: Easley & Kleinberg; Jackson, *Social and Economic Networks*; Osborne, *Game Theory*.
   - Scope: random graphs, cascades, epidemics, financial and energy networks. Undergraduate.
   - Prereq: probability (6.041 / 14.30).
   - Goals: robustness and fragility of networked systems, using dynamical systems, random graphs, optimization and game theory.

3. **MIT 6.042J Mathematics for Computer Science, Spring 2015** — https://ocw.mit.edu/courses/6-042j-mathematics-for-computer-science-spring-2015/pages/syllabus/
   - Books: an open textbook on the course site (authors not named on the syllabus page).
   - Scope: proofs, number theory, graphs, counting, probability.
   - Prereq: 18.01.
   - Goals: discrete-math methods for computer science.

4. **ETH 401-3052-10L Graph Theory, FS 2026** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=197875&semkez=2026S&ansicht=KATALOGDATEN&lang=de
   - Books: West, *Introduction to Graph Theory*; Diestel, *Graph Theory*.
   - Scope: trees, matrix-tree theorem, Menger's theorem, Euler/Hamilton cycles, matchings, planarity, colourings. 9 ECTS, master's level (includes EE&IT).
   - Prereq: able to write rigorous proofs.
   - Goals: overview of fundamental questions; use proof techniques independently.

5. **U. de Chile MA5505 Teoría de Grafos (since 2012)** — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=13298
   - Books: Diestel (GTM 173); Bollobás, *Modern Graph Theory*; West.
   - Scope: bridges, trees, bipartite graphs, density and distances, graph minors.
   - Prereq: MA3705 Algoritmos Combinatoriales.
   - Goals: introduce basic graph theory.

Graph and network topology also appears inside other syllabi: MIT 18.06 (graphs and networks), USP PSI3211 ("redes de bipolos e grafos") and UTN FRM Teoría de los Circuitos I (circuit topology).

#### Book ranking

| Book | Cites | Universities |
|---|---|---|
| Easley & Kleinberg | 2 | Stanford, MIT |
| West | 2 | ETH, UChile |
| Diestel | 2 | ETH, UChile |
| Newman | 1 | MIT (R) |
| Barabási | 1 | Stanford |
| Bollobás | 1 | UChile |
| Hamilton | 1 | Stanford |
| Jackson | 1 | MIT |

---

### 4. Complex variables

#### Syllabi consulted

1. **MIT 18.04, Spring 2018** — https://ocw.mit.edu/courses/18-04-complex-variables-with-applications-spring-2018/pages/syllabus/
   - Books: no required text. Rec: Brown & Churchill 9e (2013). Free online: Taylor; Beck, Marchesi, Pixton & Sabalka, *A First Course in Complex Analysis*.
   - Scope: analytic functions, Cauchy's theorem and integral formula, Taylor/Laurent series, harmonic functions, Laplace and Fourier transforms.
   - Prereq: 18.02 and 18.03.
   - Goals: key theorems plus engineering applications.

2. **ETH 401-0302-10L Komplexe Analysis (EE&IT bachelor), FS 2026** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=199386&semkez=2026S&ansicht=KATALOGDATEN&lang=de
   - Books: Brown & Churchill 9e; Needham, *Visual Complex Analysis* (2023); Kreyszig, *Advanced Engineering Mathematics* 10e; Dyke, *Laplace Transforms and Fourier Series* 2e.
   - Scope: Cauchy-Riemann equations, complex integration, residues, Taylor/Laurent series, Fourier/Laplace transforms. 4 ECTS.
   - Prereq: Analysis 1; Analysis 2 taken alongside.
   - Goals: apply complex-analysis tools and series; analyze with the Fourier transform.

3. **TU Delft EE2M11 Complexe Functietheorie** — https://etv.tudelft.nl/en-US/education/default/course?id=513 (ETV list)
   - Books: Brown (& Churchill), *Complex Variables and Applications*.
   - Scope, prereq, goals: n/s.

4. **U. de Chile MA2002 Cálculo Avanzado y Aplicaciones** (3 weeks on complex variables) — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=13266
   - Books (list): Brown & Churchill, *Variable Compleja y Aplicaciones* (Spanish, 1992); Spiegel, *Variable Compleja* (Schaum); Kreyszig (Spanish); Apostol; Marsden & Tromba; O'Neil; others.
   - Scope: vector-calculus theorems, complex functions, Fourier series and transforms, PDEs, calculus of variations.
   - Prereq: MA2601, MA2001.
   - Goals: vector-field calculus for physical problems; classical PDEs.

#### Book ranking

| Book | Cites | Universities |
|---|---|---|
| Brown & Churchill | 4 | MIT (9e), ETH (9e), TU Delft, UChile (Spanish) |
| Kreyszig, *Advanced Engineering Mathematics* | 2 | ETH, UChile |
| Spiegel (Schaum) | 1 | UChile |
| Needham | 1 | ETH |
| Beck et al. (free) | 1 | MIT |

---

### 5. Electric circuits I & II

#### Syllabi consulted

1. **MIT 6.002, Spring 2007** — https://ocw.mit.edu/courses/6-002-circuits-and-electronics-spring-2007/pages/syllabus/
   - Books: R: Agarwal & Lang, *Foundations of Analog and Digital Electronic Circuits* (2005).
   - Scope: lumped circuit abstraction, digital circuits, op-amps, energy-storage elements.
   - Prereq: 8.02, 18.03.
   - Goals: circuit models for design; time- and frequency-domain analysis; lab work.

2. **Georgia Tech ECE 2040, Fall 2026** — https://syllabus.gatech.edu/sites/default/files/2026-04/ECE%202040%20Fall%202026%20Syllabus.pdf
   - Books: R: Svoboda & Dorf, *Introduction to Electric Circuits* 9e (2013).
   - Scope: KCL/KVL, nodal and mesh analysis, theorems, op-amps, transients, sinusoidal steady state, frequency response, power. 3 credits.
   - Prereq: n/s.
   - Goals: RLC by hand, nodal/mesh analysis, Thévenin/Norton, phasors, frequency response.

3. **UIUC ECE 210, Fall 2025** — https://courses.grainger.illinois.edu/ece210/fa2025/
   - Books: R: Kudeki & Munson, *Analog Signals and Systems* (2009). Rec: Lathi & Green 3e; Oppenheim & Willsky 2e.
   - Scope: circuits, phasors, Fourier and Laplace transforms, filters, AM receiver.
   - Prereq: n/s (builds on ECE 110).
   - Goals: analyze and design signal-processing systems.

4. **Stanford EE 101A Circuits I** — https://edit-summer.stanford.edu/sites/summer/files/media/file/ee_101a_syllabus_summer_2026.pdf (the file name says summer 2026; the content says Fall 2025)
   - Books: no required text. Readings from Hambley, *EE Principles and Applications* 7e; Sedra & Smith 8e; Scherz & Monk 4e.
   - Scope: circuit elements including diodes and transistors; DC/AC analysis of amplifiers. 4 units.
   - Prereq: n/s.
   - Goals: circuit theory; preparation for EE 101B/114/116/153.

5. **Purdue ECE 20002** — https://engineering.purdue.edu/ECE/Academics/Undergraduates/UGO/CourseInfo/courseInfo?courseid=725&show=true&type=undergrad
   - Books: R: Talavage & Terlep, *EE Fundamentals II* 3e (2020).
   - Scope: second-order circuits, convolution, Laplace, transfer functions, filters, amplifiers, transmission lines. 3 credits.
   - Prereq: ECE 20001 and MA 26200/26600/366.
   - Goals: Laplace-based analysis, frequency behaviour, SPICE.
   - (ECE 20001 lists "notes provided by instructor" only.)

6. **KU Leuven H01Z2A Elektrische netwerken, 2026-27** — https://onderwijsaanbod.kuleuven.be/syllabi/n/H01Z2AN.htm
   - Books: R: Irwin & Nelms, *Basic Engineering Circuit Analysis* 10e.
   - Scope: DC analysis (nodal, op-amps) and AC analysis (frequency response, filters). 3 ECTS.
   - Prereq: linear systems, complex numbers, ODEs, Ohm/Kirchhoff.
   - Goals: compute linear networks under DC and AC; transfer-function basics.

7. **ETH 227-0001-00L Netzwerke und Schaltungen I, HS 2025** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=192802&semkez=2025W&ansicht=KATALOGDATEN&lang=de
   - Books: Albach, *Elektrotechnik* (Pearson 2020).
   - Scope: electrostatic and current fields, DC networks, magnetic fields, induction. 4 ECTS.
   - Prereq: n/s.
   - Goals: trace voltage and current to fields; compute DC networks.

   **ETH 227-0002-00L Netzwerke und Schaltungen II, FS 2026** — https://www.vvz.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=198525&semkez=2026S&ansicht=KATALOGDATEN&lang=de
   - Books: Albach 2e (2020); Schmidt, Schaller & Martius, *Grundlagen der Elektrotechnik – Netzwerke* 2e (2014); Sedra & Smith 7e.
   - Scope: complex AC analysis, mesh/nodal methods, Fourier, switching transients, Laplace, two-ports, op-amps. 8 ECTS.
   - Prereq: n/s.
   - Goals: time- and frequency-domain network behaviour.

8. **TU Delft EE1C11 / EE1C21 Linear Circuits A/B** — https://etv.tudelft.nl/en-US/education/default/course?id=496 (ETV list); descriptions at https://microelectronics.tudelft.nl/Education/coursedetail.php?mi=71 and https://microelectronics.tudelft.nl/Education/coursedetail.php?mi=72
   - Books: Alexander & Sadiku, *Fundamentals of Electric Circuits* (listed for EE1C21).
   - Scope: Part A covers Kirchhoff's laws, Thévenin/Norton, nodal/mesh analysis, first-order transients. Part B covers second-order transients, phasors, AC power, transformers, Bode plots, filters, Laplace, two-ports. 5 EC each.
   - Prereq: n/s.
   - Goals: the listed outcomes (compute voltages, currents and power with these methods).

9. **UPM 565000134 Teoría de Circuitos (Grado en Electrónica Industrial y Automática, 2022-23)** — https://www.upm.es/comun_gauss/publico/guias/2022-23/1S/GA_56IA_565000134_1S_2022-23.pdf
   - Books (list): Fraile Mora, *Circuitos eléctricos* (Pearson 2012); Hayt, Kemmerly & Durbin, *Análisis de circuitos en ingeniería* (2012); Alexander & Sadiku, *Fundamentos de circuitos eléctricos* (2013); Pastor & Ortega (UNED).
   - Scope: DC, single-phase AC, three-phase systems, introduction to transients, lab. 4.5 ECTS, second year.
   - Prereq: none formally; complex-number arithmetic recommended.
   - Goals: fundamentals of circuit theory.

10. **UPC 820123 Circuitos y Señales (EEBE, Electrical Engineering degree, 2026)** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/820123
    - R: Hayt, Kemmerly & Durbin (Spanish 7a); Irwin, *Análisis básico de circuitos en ingeniería* (Spanish 6a); Alexander & Sadiku (Spanish 6a).
    - Rec: Dorf & Svoboda (Spanish 3a); REA *Electric Circuits Problem Solver*.
    - Scope: transients, coupled circuits, Fourier, Laplace, resonance and filters, two-ports. 6 ECTS.
    - Prereq: Sistemes Elèctrics; Càlcul Numèric-Equacions Diferencials.
    - Goals: analysis techniques in time and frequency domains; software; lab measurement.

11. **UPC 340103 Circuitos Eléctricos (EPSEVG, Electrical Engineering degree, 2026)** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/340103
    - R: Nilsson & Riedel (Spanish 7a); Alexander & Sadiku (Spanish 6a 2018); Gómez Expósito, exercises; Nahvi & Edminister (Spanish 4a); two PSpice books.
    - Scope: unbalanced three-phase, non-sinusoidal periodic excitation, transients, state equations, Laplace. 6 ECTS.
    - Prereq: Sistemas Eléctricos recommended.
    - Goals: unbalanced Y/Δ systems, transient and frequency analysis, two-ports, simulation.

12. **UTN FRM Teoría de los Circuitos I (Electronic Eng., program dated 2008)** — http://www1.frm.utn.edu.ar/circuitos1/Apuntes/programa_TCIRC%20I.pdf
    - Books (list): Boylestad, *Introducción al análisis de circuitos* 10a (2004); Cunningham & Stuller; Dorf & Svoboda (Spanish, Alfaomega 2000); Hayt & Kemmerly, *Análisis de circuitos en ingeniería* (1985); Pueyo & Marco; Scott; Van Valkenburg, *Análisis de redes*.
    - Scope: mesh/nodal analysis, one- and two-port networks, transients, single- and three-phase AC, coupled inductors, Fourier and Laplace. 6 h/week, annual.
    - Prereq: n/s.
    - Goals: linear circuits as linear systems, in time and frequency domains.

13. **UTN FRC Teoría de los Circuitos I (Electronic Eng.)** — https://www.profesores.frc.utn.edu.ar/electronica/teoriadeloscircuitosi/
    - Books (list): Pueyo & Marco; Dorf & Svoboda; Edminister (Schaum); Skilling; Nilsson; Van Valkenburg; Balabanian et al.; Kuo; Chen.
    - Scope: time, frequency and Laplace domains; phasors, power, polyphase systems.
    - Prereq: n/s.
    - Goals: solve basic low-frequency linear-circuit problems.

14. **FIUBA 85.02 Electrotecnia (Electrical Eng., 1C/2018)** — https://cms.fi.uba.ar/uploads/8502_4f6f5e89f5.pdf
    - General recommended: Spinadel, *Circuitos eléctricos y magnéticos* 2a; Skilling; Edminister (Schaum); Fraile Mora, *Máquinas eléctricas* 5a (chapter 1); Fitzgerald et al. 6e (chapter 1).
    - Rec: Hayt, Kemmerly & Durbin (Spanish 7a); Usaola; Nilsson & Riedel (Spanish 7a); Nahvi & Edminister; Zeveke & Ionkin; Svoboda & Dorf 9e.
    - Scope: sinusoidal regime and power, resonance, topology, mesh/nodal analysis, polyphase systems (Aron method), coupled circuits, harmonics, magnetic circuits.
    - Prereq: n/s (assumes Física and Introducción a la Ingeniería Electricista).
    - Goals: consolidate electric and magnetic circuit principles for later work on machines and the grid.

15. **U. de Chile EL3101 Análisis y Diseño de Circuitos Eléctricos (from 2021)** — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=175018
    - R: Thomas & Rosa 4e (2004); Dorf & Svoboda (Spanish 6a); Nilsson & Riedel (Spanish 7a).
    - Rec: Alexander & Sadiku (Spanish); Johnson, Hilburn, Johnson & Scott 5a; Irwin (Spanish 5a).
    - Scope: resistive and dynamic circuits, Laplace, steady state including balanced three-phase. 6 credits.
    - Prereq: MA2002, FI2002.
    - Goals: analyze LTI circuits; simulate and design op-amp circuits.

16. **USP PSI3211 Circuitos Elétricos I** — https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PSI3211
    - Books (list): Orsini & Consonni vols I–II; Nilsson & Riedel 9e; Dorf & Svoboda 8e; Mariotto; Alexander & Sadiku (Portuguese 5a); Irwin & Wu 8e; Chua, Desoer & Kuh; Orsini, exercises.
    - Scope: bipoles, Kirchhoff's laws with graph topology, phasors, nodal analysis, Thévenin/Norton, first- and second-order circuits. 60 h.
    - Prereq: not shown.
    - Goals: systematic steady-state analysis; introduction to dynamics.

#### Book ranking

| # | Book | Cites | Universities (edition / language) |
|---|---|---|---|
| 1 | Dorf & Svoboda, *Introduction to Electric Circuits* | 6 | GT (9e R), UTN (Spanish, FRM + FRC), FIUBA (9e), UChile (Spanish 6a R), UPC (Spanish 3a), USP (8e) |
| 2 | Nilsson & Riedel, *Electric Circuits* | 5 | UTN FRC (Spanish), FIUBA (Spanish 7a), UChile (Spanish 7a R), UPC (Spanish 7a R), USP (9e) |
| 2 | Alexander & Sadiku, *Fundamentals of Electric Circuits* | 5 | TU Delft, UPM (Spanish), UPC (Spanish 6a R), UChile (Spanish), USP (Portuguese 5a) |
| 4 | Hayt, Kemmerly & Durbin, *Engineering Circuit Analysis* | 4 | UTN FRM (Spanish), FIUBA (Spanish 7a), UPM (Spanish 2012), UPC (Spanish 7a R) |
| 4 | Irwin (& Nelms), *Basic Engineering Circuit Analysis* | 4 | KU Leuven (10e R), UChile (Spanish 5a), UPC (Spanish 6a R), USP (8e) |
| 6 | Edminister / Nahvi & Edminister (Schaum) | 3 | UTN, FIUBA, UPC |
| 7 | Skilling | 2 | UTN, FIUBA |
| 7 | Sedra & Smith (electronics) | 2 | Stanford, ETH |

Cited once each: Albach (ETH); Boylestad, Spanish (UTN FRM); Fraile Mora, *Circuitos* (UPM); Thomas & Rosa (UChile); Van Valkenburg (UTN); Pueyo & Marco (UTN); Agarwal & Lang; Kudeki & Munson; Talavage & Terlep; Orsini & Consonni; Spinadel.

---

### 6. Electromagnetism / Physics II

#### Syllabi consulted

1. **MIT 8.02, Spring 2007** — https://ocw.mit.edu/courses/8-02-physics-ii-electricity-and-magnetism-spring-2007/pages/syllabus/
   - Books: no required text. Rec: Serway & Jewett; Tipler & Mosca; Giancoli; Young & Freedman; Resnick, Halliday & Krane.
   - Scope: electric and magnetic fields, forces, phenomena.
   - Prereq: 8.01.
   - Goals: describe, represent mathematically and predict electromagnetic phenomena from a few laws.

2. **MIT 6.013 Electromagnetics & Applications, Spring 2009** — https://ocw.mit.edu/courses/6-013-electromagnetics-and-applications-spring-2009/pages/syllabus/
   - Books: course notes. Optional: Staelin, Morgenthaler & Kong, *Electromagnetic Waves*.
   - Scope: fields, waves, radiation, transmission lines.
   - Prereq: 18.01, 18.02, 8.02, 6.002.
   - Goals: electromagnetic theory from Maxwell's equations, applied to communications, motors and sensors.

3. **TU Delft EE1P21 Electricity & Magnetism** — https://etv.tudelft.nl/en-US/education/default/course?id=507 (ETV)
   - Books: Wolfson, *Essential University Physics* vols 1–2; Müller-Kirsten, *Electrodynamics*.
   - Scope, prereq, goals: n/s.

4. **UPC 820005 Física II: Fundamentos de Electromagnetismo (EEBE, includes the Electrical Engineering degree, 2026)** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/820005
   - Books (R): Tipler & Mosca, *Física para la ciencia y la tecnología* vol 2 (Spanish 6a, 2010); two problem books (Alcaraz et al.; Alarcón et al.).
   - Scope: fields and potential, conductors and dielectrics, DC/AC current, magnetic fields, induction, Maxwell's equations. 6 ECTS.
   - Prereq: none.
   - Goals: working method plus basic electromagnetic principles for engineering problems.

5. **FIUBA 82.06 Electromagnetismo (Electronic Eng., 1C/2020)** — https://cms.fi.uba.ar/uploads/8206_7615e0fbaf.pdf
   - Books: R: course notes. Rec: Fernández (EUDEBA 2013); Ramo, Whinnery & Van Duzer 3e; Orfanidis (free online); Sadiku, *Elementos de electromagnetismo* (Spanish 2a); Zahn; Vanderlinde; Stratton.
   - Scope: Maxwell's equations, statics, magnetic materials, transmission lines, waves, antennas, numerical methods.
   - Prereq: n/s.
   - Goals: electromagnetic theory for electronic engineering; know the limits of models; simulation.

6. **U. de Chile FI2002 Electromagnetismo (from 2025)** — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=248822
   - Books (R): Griffiths, *Introduction to Electrodynamics* 4e; Cordero; Serway & Jewett vol 2 (Spanish 7a); Sapone.
   - Scope: electrostatics, currents, magnetostatics, induction, electromagnetic waves. 6 credits, 4th semester.
   - Prereq: MA2001, MA2601, FI2001.
   - Goals: characterize electromagnetic phenomena up to energy propagation.

7. **USP 4323203 Física III** — https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=4323203
   - Books (list): Young & Freedman, *Física III* (Portuguese 12a, 2009); Nussenzveig vol 3; Feynman Lectures vol 2.
   - Scope: charge through induction and magnetic materials. 60 h.
   - Prereq: not shown.
   - Goals: basic electricity and magnetism for engineers.

(ETH 227-0001 covers fields inside Netzwerke und Schaltungen I, using Albach.)

#### Book ranking

| Book | Cites | Universities |
|---|---|---|
| Serway & Jewett | 2 | MIT, UChile (Spanish 7a R) |
| Tipler & Mosca | 2 | MIT, UPC (Spanish 6a R) |
| Young & Freedman (Sears-Zemansky) | 2 | MIT, USP (Portuguese 12a) |
| Griffiths | 1 | UChile (R) |
| Halliday / Resnick / Krane | 1 | MIT |
| Giancoli | 1 | MIT |
| Wolfson | 1 | TU Delft |
| Sadiku, *Elements* | 1 | FIUBA |
| Ramo, Whinnery & Van Duzer | 1 | FIUBA |
| Nussenzveig | 1 | USP |
| Feynman vol 2 | 1 | USP |

---

### 7. Electric machines & transformers

#### Syllabi consulted

1. **MIT 6.685 Electric Machines, Fall 2013** — https://ocw.mit.edu/courses/6-685-electric-machines-fall-2013/pages/syllabus/
   - Books: course notes. Rec: Fitzgerald, Kingsley & Umans 6e; Kirtley, *Electric Power Principles* (2010); Beaty & Kirtley, *Electric Motor Handbook*.
   - Scope: transformers, transducers, DC, induction and synchronous machines, dynamics. Graduate.
   - Prereq: 6.061/6.690.
   - Goals: design rotating and linear machines; estimate dynamic parameters.

2. **MIT 6.061 Intro to Electric Power Systems, Spring 2011** — https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/pages/syllabus/
   - Books: R: Kirtley, *Electric Power Principles*.
   - Scope: magnetic circuits, synchronous/induction/DC machines, power electronics, system operation.
   - Prereq: 6.002, 6.013.
   - Goals: model and interconnect power apparatus with lumped parameters and transformations.

3. **Purdue ECE 32100, Fall 2015** — https://engineering.purdue.edu/~dionysis/321_F15_syllabus.pdf
   - Books: R: Krause, Wasynczuk & Pekarek, *Electromechanical Motion Devices* 2e (2012).
   - Scope: energy conversion; DC, induction, brushless DC and stepper motors.
   - Prereq: ECE 20200, electricity & magnetism physics, ECE 25500.
   - Goals: analyze electromagnetic devices, reference-frame transformations, machines and DC drives.

4. **UIUC ECE 330, Fall 2025** — https://ws.engr.illinois.edu/custom/getsyllabus.asp?id=3130
   - Books: Pai, *Power Circuits and Electromechanics* (Stipes).
   - Scope: complex power, three-phase, magnetic circuits, transformers, energy/co-energy, synchronous and induction machines, dynamics and stability.
   - Prereq: n/s. Goals: n/s.

5. **TU Delft EE2E11 Electrical Energy Conversion** — https://etv.tudelft.nl/en-US/education/default/course?id=509 (ETV)
   - Books: Ferreira, *The Principles of Electromechanical Power Conversion*.
   - Scope, prereq, goals: n/s.

6. **KU Leuven H01L8A Elektrische energie en aandrijvingen, 2026-27** — https://onderwijsaanbod.kuleuven.be/syllabi/n/H01L8AN.htm
   - Books: textbook sold through Acco, title not named.
   - Scope: three-phase, transformers, DC/induction/synchronous machines, power quality. 6 ECTS.
   - Prereq: n/s.
   - Goals: practical understanding of energy systems and machines.

7. **UPC 820127 Máquinas Eléctricas I (EEBE, Electrical Engineering degree, 2026)** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/820127
   - R: Fitzgerald & Umans 7e (2014); Fraile Mora 7a (2015).
   - Rec: Boldea; Gross; Sen 3e.
   - Scope: transformers; AC-machine transient models. 6 ECTS.
   - Prereq: Sistemes Elèctrics; calculus, matrix calculus.
   - Goals: electromechanical conversion, transformers and induction machines.

8. **UPC 340102 Máquinas Eléctricas I (EPSEVG, 2026)** — https://www.upc.edu/content/grau/guiadocent/pdf/esp/340102/maquinas-electricas-i.pdf
   - R: Chapman, *Máquinas eléctricas* 5a (2012); Fitzgerald, Kingsley & Umans (Spanish 6a).
   - Rec: Fraile Mora 8a (2016); Sanz Feito.
   - Scope: single- and three-phase transformers, energy conversion, synchronous machines. 150 h.
   - Prereq: n/s.
   - Goals: equivalent circuits, steady state, lab testing.

9. **FIUBA 85.36 Máquinas Eléctricas (Electrical Eng., 1C/2018; same list in 65.06)** — https://cms.fi.uba.ar/uploads/8536_bd32c94704.pdf (and https://cms.fi.uba.ar/uploads/6506_822bb055fe.pdf)
   - Rec (for the course): notes; Chapman; Guru & Hiziroglu; Wildi 6a.
   - Reference: Fitzgerald, Kingsley & Umans (Spanish 5a); MIT EE Staff, *Circuitos magnéticos y transformadores*; Hindmarsh; Cathey.
   - Scope: transformers, induction, synchronous, DC and special machines, heating, switchgear.
   - Prereq (stated as required knowledge): Electrotecnia General, calculus, physics.
   - Goals: construction, operation, selection and control of machines.

10. **UTN FRBA Máquinas Eléctricas I (950528, Electrical Eng., 2017)** — https://frba.utn.edu.ar/wp-content/uploads/2018/06/M%C3%A1quinas-El%C3%A9ctricas-I-Ing.-El%C3%A9ctrica-950528.pdf
    - R: Kostenko & Piotrovsky, *Máquinas Eléctricas* vol I; Gourishankar, *Conversión de energía electromecánica*; Cortes Cherta; MIT Staff, *Circuitos magnéticos y transformadores*.
    - Rec: Lemozy notes.
    - Scope: single- and three-phase transformers, transients, conversion principle, DC machines. 144 h, annual.
    - Prereq: n/s.
    - Goals: model transformers and DC machines analytically, graphically and numerically; industrial selection.

11. **UNAM FI 1656 Máquinas Eléctricas I (approved 2005)** — Wayback copy: http://web.archive.org/web/2023id_/https://www.ingenieria.unam.mx/programas_academicos/licenciatura/Electrica_Electronica/06/maquinas_electricas_i.pdf (live URL now 404)
    - Books (list): El-Hawary; Kosow; Langsdorf; Matsch; Puchstein; Sen 2e; Pérez Amador; Chapman (Spanish 1998); Fitzgerald, Kingsley & Kusko (2004).
    - Scope: magnetic circuits, transformers, induction, synchronous and DC machines, AC motor control. 104 h, 11 credits.
    - Prereq: Análisis de Circuitos Eléctricos.
    - Goals: qualitative and quantitative analysis of machines; plan installation and operation.

12. **U. de Chile EL4111 Conversión de la Energía y Sistemas Eléctricos (from 2022)** — https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=128814
    - R: Fitzgerald, Kingsley & Umans (Spanish 6a); Romo & Vargas notes; Brokering, Palma & Vargas, *Atrapando el Sol* (free online).
    - Rec: Sanz; Gómez Expósito; Nasar; Gourishankar; Langsdorf; Chapman 4a; Glover & Sarma; Saadat; Grainger & Stevenson; others.
    - Scope: conversion fundamentals, transformers, DC and AC machines, generation. 6 credits.
    - Prereq: EL3101.
    - Goals: evaluate electrical, magnetic and mechanical behaviour with analytic and circuit models.

13. **USP PEA3306 Conversão Eletromecânica de Energia** — https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PEA3306
    - Books (list): Falcone, *Eletromecânica*; Umans, *Máquinas Elétricas de Fitzgerald e Kingsley* (Portuguese 7a); Chapman, *Fundamentos de máquinas elétricas* (2013).
    - Scope: transformers, converters, rotating fields, synchronous, induction and DC machines. 60 h.
    - Prereq: not shown.
    - Goals: understand, model and analyze electromechanical devices.

Relevant to ToOp though outside this scope: UChile EL4103 (power systems; https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=124166) cites Gómez Expósito, Wood & Wollenberg 3e, Grainger & Stevenson and Saadat. ETH 227-0122 (power transmission, including power flow) uses a lecture script only.

#### Book ranking

| # | Book | Cites | Universities (edition / language) |
|---|---|---|---|
| 1 | Fitzgerald, Kingsley & Umans, *Electric Machinery* | 6 | MIT (6e), UPC (7e R, Spanish 6a R), FIUBA (Spanish 5a), UNAM, UChile (Spanish 6a R), USP (Portuguese 7a) |
| 2 | Chapman, *Electric Machinery Fundamentals* | 5 | FIUBA, UNAM (Spanish), UPC (Spanish 5a R), UChile (Spanish 4a), USP (Portuguese) |
| 3 | Fraile Mora, *Máquinas eléctricas* (Spanish) | 2 | UPC (7a R, 8a), FIUBA (Electrotecnia course, chapter 1). Also a UPM design course. |
| 4 | Gourishankar | 2 | UTN (R), UChile |
| 4 | MIT EE Staff, *Magnetic Circuits and Transformers* | 2 | UTN (R), FIUBA |
| 4 | Sen | 2 | UNAM, UPC |
| 4 | Langsdorf | 2 | UNAM, UChile |
| 4 | Sanz Feito | 2 | UPC, UChile |

Cited once each: Kirtley (MIT, two courses); Kostenko & Piotrovsky (UTN R); Krause (Purdue); Pai (UIUC); Ferreira (TU Delft); Wildi; Guru & Hiziroglu.

---


## Part 2 — Power systems, optimization and computing

### 8. Power system analysis
#### Syllabi consulted

**US**
- **Texas A&M ECEN 460 Power System Operation and Control, Spring 2025 (Overbye)**
  URL: https://overbye.engr.tamu.edu/wp-content/uploads/sites/146/2025/01/ECEN460_Syllabus_Spring2025_Public.pdf
  - Scope: undergraduate, 4 credits with lab. Complex power, 3-phase, per-unit, component models, Ybus, power flow and solution methods, power-flow sensitivity, contingency analysis, economic dispatch, OPF/SCOPF/LMP, markets, stability, controls, renewables. PowerWorld labs.
  - Prerequisites: ECEN 215 or 314 (ECEN 340 recommended).
  - Goals: n.s.
  - Books: R Glover, Overbye, Birchfield & Sarma, *Power System Analysis and Design*, 7th ed., Cengage 2023. The Spring 2026 offering (Birchfield) requires the same book: https://birchfield.engr.tamu.edu/460s26
- **UIUC ECE 476 Power System Analysis (catalog)**
  URL: https://ece.illinois.edu/academics/courses/ece476
  - Scope: 3 credit hours. 3-phase, per-unit, line parameters, power flow, symmetrical components and short circuit, transient stability, economic dispatch, relaying.
  - Prerequisites: ECE 330.
  - Goals: overview of interconnected operation; build system models; do power flow, economic dispatch and short circuit; write a basic power-flow program.
  - Books: Glover, Overbye & Sarma, 6th ed. The Fall 2011 page lists the 5th ed.: https://courses.grainger.illinois.edu/ece476/fa2011/
- **Georgia Tech ECE 4320 Power System Analysis and Control**
  URL: https://ece.gatech.edu/courses/ece4320
  - Scope: power flow, OPF, unit commitment, state estimation, voltage/frequency control, economic dispatch.
  - Prerequisites: ECE 3072.
  - Goals: formulate and solve power-flow problems; assess controls; solve economic dispatch.
  - Books: the textbook field gives only generic titles, no authors. Not counted.
- **UC Berkeley EE 137A Introduction to Electric Power Systems, Fall 2017 (von Meier)**
  URL: https://tbp.studentorg.berkeley.edu/syllabi/797/download
  - Scope: 4 units. System design, generators, transmission and distribution components, power flow, operation.
  - Prerequisites: Math 54, Physics 7B, EE 16AB.
  - Goals: explain component functions and limits; quantitative AC analysis; power-flow methods in PowerWorld; P/Q balance, stability and security.
  - Books: R Bergen & Vittal, *Power Systems Analysis*, 2nd ed., 2000. Rec von Meier, *Electric Power Systems: A Conceptual Introduction* (2006). Masters, *Renewable and Efficient Electric Power Systems*, 2nd ed. (required in 137B).
- **MIT 6.061/6.690 Introduction to Electric Power Systems, Spring 2011**
  URL: https://ocw.mit.edu/courses/6-061-introduction-to-electric-power-systems-spring-2011/pages/syllabus/
  - Scope: advanced undergraduate. Energy circuits, magnetics, machines, power electronics, interconnection.
  - Prerequisites: 6.002, 6.013.
  - Goals: n.s.
  - Books: Kirtley, *Electric Power Principles*, Wiley 2010 (free OCW notes).
- **MIT 6.691 Seminar in Electric Power Systems, Spring 2006**
  URL: https://ocw.mit.edu/courses/6-691-seminar-in-electric-power-systems-spring-2006/pages/syllabus/
  - Scope: graduate, variable topics (economics/deregulation, DC distribution, renewables).
  - Prerequisites: 6.061 or network theory plus controls/EM.
  - Goals: n.s.
  - Books: Bergen & Vittal, 2nd ed.
- **Cornell ECE 4510 Electric Power Systems I, Fall 2025 roster**
  URL: https://classes.cornell.edu/browse/roster/FA25/class/ECE/4510
  - Scope: 3 credits. Line/transformer/generator models, per-unit, network matrices, power flow, voltage control, economic dispatch.
  - Prerequisites: ECE 3250.
  - Goals: component models; power flow and economic dispatch; response to contingencies.
  - Books: none listed.
- **Stanford CEE 272R Modern Power Systems Engineering, 2019.** SECONDARY SOURCE ONLY: a Course Hero copy; the official syllabus was not accessible. Books: Glover et al., 5th ed.; Kirschen & Strbac; von Meier (optional). Not counted.

**Europe**
- **KU Leuven H04A9A Power System Calculations**
  URL: https://onderwijsaanbod.kuleuven.be/syllabi/e/H04A9AE.htm
  - Scope: master, 3 ECTS. Component models, power flow, OPF/ED/UC, short circuits.
  - Prerequisites: H01L8A (basic AC/energy).
  - Goals: derive models; apply them in power flow and OPF by hand and in software.
  - Books (Rec): Grainger & Stevenson; Glover, Sarma & Overbye.
- **UCLouvain LELEC2520 Electrical power systems**
  URL: https://uclouvain.be/en-cours-2025-lelec2520
  - Scope: master, 5 credits. Architecture, steady-state modelling, calculation, optimization.
  - Prerequisites: LELEC1370, LELEC1310.
  - Goals: outcome codes only.
  - Books: reference textbook Gómez-Expósito, Conejo & Cañizares, *Electric Energy Systems: Analysis and Operation*.
- **ULiège ELEC0447 Analysis of electric power and energy systems**
  URL: https://www.programmes.uliege.be/cocoon/20252026/en/cours/ELEC0447-1.html
  - Scope: master, 5 credits. Per-unit, components, power-flow solvers, frequency/voltage control, stability.
  - Prerequisites: circuits, programming, numerical methods.
  - Goals: understand operation and its mathematical/numerical modelling.
  - Books: mainly Mohan, *Electric Power Systems: A First Course*, Wiley 2012.
- **ETH 227-0526-00L Power System Analysis (Hug, Valverde)**
  URL: https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=192800&semkez=2025W&ansicht=ALLE&lang=en
  - Scope: MSc, 6 ECTS. Stationary models, Newton-Raphson power flow, short circuit, symmetrical components, distribution power flow, state estimation.
  - Prerequisites: n.s.
  - Goals: stationary dependencies and analysis tools in normal and faulted states.
  - Books: lecture notes only.
- **DTU 46705 Power grid analysis**
  URL: https://kurser.dtu.dk/course/46705
  - Scope: MSc, 5 ECTS. Power flow in Python, faults.
  - Prerequisites (recommended): 31730/46700.
  - Books: none.
- **KTH EG2100 Power System Analysis**
  URL: https://www.kth.se/student/kurser/kurs/EG2100?l=en
  - Scope: load flow in Matlab, symmetrical/unsymmetrical steady state.
  - Prerequisites: linear algebra, calculus, Matlab, EJ1200.
  - Books: literature field empty.
- **Edinburgh ELEE09029 Power Systems, Power Electronics and Machines 3**
  URL: http://www.drps.ed.ac.uk/25-26/dpt/cxelee09029.htm
  - Scope: undergraduate year 3. Load flow, faults, machines, converters.
  - Prerequisites: SCEE08008.
  - Goals: load flow and fault calculations; machines and converters.
  - Books: Saadat, *Power System Analysis*; Mohan, Undeland & Robbins; Chapman.
- **Edinburgh ELEE10032 Power Systems 4**
  URL: http://www.drps.ed.ac.uk/25-26/dpt/cxelee10032.htm
  - Scope: year 4. Generator dq model, short circuits, protection, distributed generation.
  - Prerequisites: ELEE09029 recommended.
  - Books: Chapman; Grainger & Stevenson.
- **Chalmers ENM125 Sustainable electric power systems**
  URL: https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/ENM125/?acYear=2025%2F2026
  - Scope: master, 7.5 credits, for students without an electrical background.
  - Books: Glover, Overbye & Sarma, 6th ed. SI; Powell.
- **Sevilla 2030104 Sistemas Eléctricos de Potencia, 2024-25 (taught by Gómez-Expósito)**
  URL: POST form at https://sevius4.us.es/index.php?PyP=LISTA&codcentro=45&titulacion=203&asignatura=2030104
  - Scope: industrial engineering degree year 4, 6 ECTS. Per-unit, Gauss-Seidel/Newton-Raphson/fast decoupled/DC power flow, static security analysis, short circuit, transient stability.
  - Prerequisites: n.s.
  - Goals: know system structure, real-time operation and planning; simulate a real network.
  - Books (Rec, all Spanish):
    - Gómez-Expósito (coord.), *Análisis y operación de sistemas de energía eléctrica*, McGraw-Hill 2002
    - Barrero, *Sistemas de Energía Eléctrica*, Thomson 2004
    - Grainger & Stevenson, *Análisis de Sistemas Eléctricos de Potencia*, McGraw-Hill 1996
    - Gómez-Expósito et al., *Problemas resueltos*, 2003
- **Comillas ICAI DIE-GITI-323 Sistemas de Energía Eléctrica, 2020-21**
  URL: https://repositorio.comillas.edu/server/api/core/bitstreams/ac4b2e20-7d2b-48b4-a89e-a4fdf7f19e6d/content
  - Scope: year 3, 6 ECTS. Components, power flow, Q-V and P-f control, state estimation.
  - Goals: pick the right model per analysis; economic operation.
  - Books: R Gómez-Expósito, Conejo & Cañizares (CRC). Rec Wood & Wollenberg; Elgerd.
  - The older Comillas AES10 guide (https://repositorio.comillas.edu/server/api/core/bitstreams/df1077a3-faae-41b0-b7e6-3b2c7301177b/content) lists R Elgerd 1983; Grainger & Stevenson (ES); Gómez-Expósito 2002 (ES). Rec Kundur.

**Latin America**
- **FIUBA 6515 Sistemas Eléctricos de Potencia, 2015**
  URL: https://cms.fi.uba.ar/uploads/6515_ad5e1ddb96.pdf
  - Scope: per-unit, load flow (Gauss-Seidel/Newton-Raphson, network matrices), voltage/FACTS, faults, economic operation, stability, HVDC.
  - Prerequisites: n.s.
  - Goals: apply line/machine parameters, load flow, faults and stability.
  - Books: R Grainger & Stevenson (EN); Saadat 2002; Glover-Sarma-Overbye 2012; Kundur; Padiyar; Sauer & Pai; Greenwood; Wood & Wollenberg 1996; others. Rec Gross; Bergen & Vittal; Weedy; Elgerd; Kimbark.
- **UTN FRBB Sistemas de Potencia, 2025–26**
  URL: https://www.frbb.utn.edu.ar/frbb/info/departamentos/electrica/programas_analiticos/sistemas_de_potencia.pdf
  - Scope: 128 h. Lines, power flow, faults, stability, economic dispatch.
  - Prerequisites: Materiales, Máquinas I, Electrotecnia II.
  - Goals: basic knowledge; solve small real problems.
  - Books: Grainger & Stevenson, *Análisis de Sistemas de Potencia* (ES, 1998); Elgerd; Gross (ES); Greenwood; course notes.
- **UTN FRM Centrales y Sistemas de Transmisión**
  URL: http://www1.frm.utn.edu.ar/cstransmision/programa.html
  - Prerequisites and goals: n.s.
  - Books: Grainger & Stevenson (ES); Weedy (ES); others.
- **UNR E21 Sistemas de Potencia, Plan 2014**
  URL: https://web.fceia.unr.edu.ar/images/PDF/Webs_de_asignaturas/El%C3%A9ctrica/E21-Sistemas_de_Potencia.pdf
  - Scope: 9th semester, 80 h. Power flow, OPF (hydrothermal), HVDC, P/Q control, stability.
  - Prerequisites: E14, E17.
  - Goals: static analysis with computer tools; operation and planning with economic implications.
  - Books: R Kundur; Arrillaga & Watson; El-Hawary. Rec Glover & Sarma, *Sistemas de Potencia: Análisis y Diseño*, 3ª ed., Thomson 2003 (ES); Kothari & Nagrath (ES); Wood, Wollenberg & Sheblé; Vaahedi; Barrero.
- **UNLP E1239 Sistemas de Potencia, Plan 2018**
  URL: https://www1.ing.unlp.edu.ar/sitio/academica/asignaturas/asignatura.php?cod=E1239
  - Scope: year 5, 96 h. Argentine wholesale market, power flow, voltage control, optimal dispatch with nodal factors, faults, transient stability and contingency evaluation. PSS/E, ATP and Matlab.
  - Prerequisites: E1237, E1234, E1235, English.
  - Goals: large interconnected systems; models applied to near-real cases.
  - Books: Glover-Sarma-Overbye 6th ed. 2017; Gómez-Expósito 2002 (ES); Kothari & Nagrath (ES); Stevenson & Grainger (ES 1996); Stevenson (ES); Weedy (ES); Elgerd; Stagg & El-Abiad; Kundur; many others.
- **UNLP E1235 Teoría de la Transmisión**
  URL: https://www1.ing.unlp.edu.ar/sitio/academica/asignaturas/asignatura.php?cod=E1235
  - Scope: year 4. Line parameters, per-unit, faults, matrix network methods, HVDC.
  - Prerequisites: E1233, E1204.
  - Books: similar list (Glover, Grainger & Stevenson ES, Gómez-Expósito ES, Kothari & Nagrath, and others).
- **U. de Chile EL4103 Sistemas de Energía y Equipos Eléctricos, 2022**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=124166
  - Scope: 6 credits. Models, power flow (iterative and decoupled), voltage control, faults, economic operation.
  - Prerequisites: EL4111.
  - Goals: analyse normal and faulted operation; optimization for economic operation.
  - Books: Brokering & Palma (ES); Gómez-Expósito 2003 (ES); Wood & Wollenberg 2013; Grainger & Stevenson 2ª ed. (ES); Saadat.
- **PUC Chile IEE2312 Sistemas de Potencia**
  URL: https://catalogo.uc.cl/index.php?tmpl=component&option=com_catalogo&view=programa&sigla=IEE2312
  - Scope: 10 credits.
  - Prerequisites: IEE1122.
  - Goals: operation under technical and economic limits.
  - Books: Grainger & Stevenson 1994 (EN); Gönen; Brokering (ES); Almeida & Freitas (PT).
- **USP PEA3410 / PEA3417 Sistemas de Potência I and II**
  URLs: https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PEA3410 and ?sgldis=PEA3417
  - Scope: line models, nodal matrices, short circuit, power flow, redispatch and load shedding for overloads.
  - Prerequisites: PEA3301.
  - Goals: model and analyse steady-state networks.
  - Books: Grainger & Stevenson 1994; Zanetta (PT); El-Hawary; Oliveira-Schmidt-Kagan-Robba (PT); Stagg & El-Abiad; Weedy (PT).
- **UNICAMP ET720 / ET931, catalog 2026**
  URL: https://www.dac.unicamp.br/sistemas/catalogos/grad/catalogo2026/disciplinas/et.html
  - Scope: 60 h each. ET931 covers Newton and decoupled load flow and sparse storage.
  - Prerequisites: ET620 / ET720.
  - Goals: n.s.
  - Books: Monticelli & Garcia (PT); Monticelli, *Fluxo de Carga* (PT); Stevenson, *Elements*; Elgerd (PT); Robba (PT).
- **UNAM Especialización, Análisis de Sistemas Eléctricos**
  URL: https://www.ingenieria.unam.mx/puei/asignaturas/energiaelectrica/anadsistelec.pdf
  - Scope: 64 h.
  - Prerequisites: none.
  - Goals: short circuit, power flow, harmonics and arc flash with software.
  - Books: Grainger & Stevenson 1994 (EN); Das; Anderson; Arrillaga; IEEE 399; others.
- **UNAM graduate power-systems admission syllabus** (not a course)
  URL: http://posgrado.electrica.unam.mx/temarios/potencia.pdf
  - Books: Grainger & Stevenson, *Análisis de Sistemas de Potencia* (ES 1996); Glover & Sarma, *Sistemas de Potencia: Análisis y Diseño* 3ª ed. (ES); Elgerd; Stagg & El-Abiad; Anderson.

#### Book ranking

| # | Book | Univ. | Universities | Editions / languages seen |
|---|---|---|---|---|
| 1 | Grainger & Stevenson, *Power System Analysis*; ES *Análisis de sistemas de potencia* | 12 | GT; KU Leuven, Edinburgh, Sevilla, Comillas; FIUBA, UTN, UNLP, U Chile, PUC, USP, UNAM | EN McGraw-Hill 1994; ES McGraw-Hill México 1996 (UTN, UNLP, U Chile, UNAM, Sevilla, Comillas) |
| 2 | Glover, Sarma, Overbye (+Birchfield), *Power System Analysis and Design*; ES *Sistemas de potencia: análisis y diseño* | 12 | TAMU, UIUC, GT, UT Austin, UW; KU Leuven, Chalmers, Edinburgh; FIUBA, UNR, UNLP, UNAM | 7th ed. 2023 (TAMU, GT); 6th (UIUC, Chalmers, UNLP); 5th/4th/3rd older; ES 3ª ed. Thomson 2003 (UNR, UNAM) |
| 3 | Gómez-Expósito, Conejo, Cañizares, *Electric Energy Systems: Analysis and Operation*; ES *Análisis y operación de SEE* | 8 | UT Austin; KU Leuven, UCLouvain, Comillas, Sevilla; U Chile, UNLP, USP | EN CRC 2008 / 2nd ed. 2018; ES McGraw-Hill 2002; PT LTC 2011 |
| 4 | Elgerd, *Electric Energy Systems Theory* | 6 | Comillas; FIUBA, UTN, UNLP, UNICAMP (PT), UNAM | 1981–83 |
| 5 | Bergen & Vittal, *Power Systems Analysis* | 5 | MIT, Berkeley, GT, UT Austin; FIUBA | 2nd ed. 2000 |
| 6 | Saadat, *Power System Analysis* | 4 | Edinburgh, Chalmers; FIUBA, U Chile | 1999/2002; 3rd ed. PSA 2010–11 |
| 7 | Weedy, *Electric Power Systems*; ES *Sistemas eléctricos de gran potencia* | 4 | FIUBA, UTN, UNLP, USP | ES Reverté 1978; PT |
| 8 | Stevenson, *Elements of Power System Analysis* | 3 | UNLP (ES), UNICAMP, USP | |
| 9 | Stagg & El-Abiad, *Computer Methods in Power System Analysis* | 3 | UNLP, USP, UNAM | 1968 |

- Kundur, *Power System Stability and Control* (outside ToOp's steady-state core): 8 universities (KU Leuven, Chalmers, Comillas, FIUBA, UNR, UNLP, U Chile, USP).
- Two universities: Kothari & Nagrath (ES).
- One university: Mohan (ULiège), Kirtley (MIT).
- Notes only: ETH, DTU, KTH, Cornell.


### 9. Power system operation and control, security, OPF, congestion
#### Syllabi consulted

**US**
- **UIUC ECE 573 Power System Operations (catalog, updated 2013)**
  URL: https://ece.illinois.edu/academics/courses/ece573
  - Scope: graduate. EMS, security analysis, OPF/SCOPF, unit commitment (DP, Lagrangian relaxation), state estimation, restructuring, congestion management.
  - Prerequisites: ECE 476; ECE 530 concurrent.
  - Goals: n.s.
  - Books: class notes. Rec (not required) Wood & Wollenberg, 2nd ed. 1996; Monticelli, *State Estimation in Electric Power Systems*, Kluwer 1999.
- **Georgia Tech ECE 6320 Power System Operation and Control, Fall 2026 (Grijalva)**
  URL: https://syllabus.gatech.edu/sites/default/files/2026-03/ECE6320%20Syllabus.pdf
  - Scope: graduate, 3 credits. Security assessment, economic optimization, dynamics, DER operation.
  - Prerequisites: graduate standing; ECE 4320-level course desirable.
  - Goals: computational methods for bulk operation; security-assessment modelling; economic optimization algorithms; dynamics; EMS architecture.
  - Books: electronic notes (required). Background: Glover et al. 7th; Bergen & Vittal 2nd; Grainger & Stevenson 1994. Complementary: Wood & Wollenberg 3rd ("the Bible"); Sauer, Pai & Chow 2nd 2017; Kersting 4th.
  - The Fall 2024 version (https://bpb-us-e1.wpmucdn.com/sites.gatech.edu/dist/4/1075/files/2024/10/ECE6320_Syllabus_F2024.pdf) also adds Abur & Gómez-Expósito, *Power System State Estimation*.
- **UT Austin EE 394V Power System Operations & Control, Spring 2020 (Hao Zhu)**
  URL: https://utdirect.utexas.edu/apps/student/coursedocs/nlogon/download/10298680/
  - Scope: graduate, 3 credits. Economic dispatch, AGC, gradient/Newton methods, Lagrange for dispatch, sensitivity factors for security, state estimation.
  - Prerequisites: power flow at EE 368L/369 level, linear algebra, probability, Matlab/Python.
  - Goals: as listed in the scope.
  - Books (all Rec, none required):
    - Wood, Wollenberg & Sheblé, 3rd ed. 2014
    - Bergen & Vittal, 2nd ed.
    - Baldick, *Applied Optimization*, Cambridge UP 2006
    - Boyd & Vandenberghe
    - Gómez-Expósito, Conejo & Cañizares, 2018
    - Taylor, *Convex Optimization of Power Systems*, Cambridge UP 2015
- TAMU ECEN 460 (see section 8) also covers contingency analysis, SCOPF and LMP with Glover 7th.

**Europe**
- **KU Leuven H04C6A Design and Management of Electric Power Systems (Van Hertem)**
  URL: https://onderwijsaanbod.kuleuven.be/syllabi/e/H04C6AE.htm
  - Scope: master, 6 ECTS. PTDF/LODF, OPF, network reduction, state estimation, N-1/N-k reliability, HVDC/FACTS, planning, asset management.
  - Prerequisites: H04A0A, H04A9A.
  - Goals: from planning to real-time operation.
  - Books (Rec): Kundur 1994; Gómez-Expósito, Conejo & Cañizares, CRC 2008; Grainger & Stevenson 1994.
- **ULB ELEC-H413 Electric Power Systems I**
  URL: https://www.ulb.be/en/programme/elec-h413
  - Scope: 5 ECTS. Load flow, weighted-least-squares state estimation, economic dispatch/UC, OPF, security under contingencies.
  - Prerequisites: n.s.
  - Books: Debs, *Modern Power Systems Control and Operation*, 1988; Wood & Wollenberg, 2nd ed. 1996.
- **Chalmers ENM066 Advanced power system analysis**
  URL: https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/ENM066/?acYear=2025%2F2026
  - Scope: 7.5 credits. Economic dispatch, UC, large-scale power flow, OPF, markets, stability.
  - Prerequisites: ENM052.
  - Books: Kirschen & Strbac 2004; Saadat 3rd; Kundur; Van Cutsem & Vournas.
- **ULiège ELEC0448 Planning and operation**
  URL: https://www.programmes.uliege.be/cocoon/20252026/en/cours/ELEC0448-1.html
  - Prerequisites: physics, optimization, programming.
  - Books: slides and papers.
- **DTU 46750 Optimization in modern power systems**
  URL: https://kurser.dtu.dk/course/46750
  - Scope: convex optimization, duality, uncertainty, applied to operation and markets.
  - Prerequisites (recommended): 46705, 42101 etc., plus Python/Julia.
  - Books: none listed.
- **DTU 46770 Integrated energy grids**
  URL: https://kurser.dtu.dk/course/46770
  - Scope: AC-OPF, DC-OPF, convexification.
  - Books: none listed.
- **ETH 227-0530-00L Optimization in Energy Systems**
  URL: https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=199315&semkez=2026S&ansicht=ALLE&lang=en
  - Books: none listed.
- **KTH EG2200 Power Generation Operation and Planning**
  URL: https://www.kth.se/student/kurser/kurs/EG2200?l=en
  - Books: literature field empty.

**Latin America**
- **U. de Chile EL7020 Análisis y Operación de SEP, 2013**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=3809
  - Scope: graduate.
  - Prerequisites: EL4103.
  - Goals: solve real operation problems; choose tools.
  - Books: R Wood & Wollenberg; Billinton & Allan, *Reliability Evaluation of Power Systems*. Rec Gómez-Expósito (ES); Kundur; Grainger & Stevenson (ES).
- **UNSJ-IEE postgraduate courses, 2024** (80 h each; no books, prerequisites or goals stated)
  - Flujo de Potencia y Métodos de Optimización: https://fi.unsj.edu.ar/panel_fi/archivos/noticias/d476e432-c751-11ee-83c1-96b8d93456aa.pdf (Newton-Raphson, LP/MILP/NLP, Benders, Lagrangian relaxation, OPF)
  - Despacho Económico y Planeamiento: https://fi.unsj.edu.ar/panel_fi/archivos/noticias/d486f57a-c751-11ee-83c1-96b8d93456aa.pdf
- **USP PEA3523 Operação e Comercialização**
  URL: https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PEA3523
  - Prerequisites: PEA3420.
  - Books: Wood, Wollenberg & Sheblé 3rd 2014; Gómez-Expósito et al. (PT, LTC 2011); Kirchmayer; Brazilian market texts.
- **USP PEA3422 Métodos de Otimização Aplicados a Sistemas Elétricos**
  URL: https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PEA3422
  - Scope: LP/MILP; heuristic network reconfiguration; GA, PSO, differential evolution; multi-objective.
  - Prerequisites: n.s.
  - Goals: classical and heuristic methods for power problems.
  - Books: Hillier & Lieberman; Goldberg 1989; Kagan et al. (PT); Michalewicz 3e; Deb 2001; Di Barba.
  - This is the closest match found to ToOp's evolutionary search.
- FIUBA 6515, UNR E21 and UNLP E1239 (see section 8) also cite Wood & Wollenberg.

#### Book ranking

| # | Book | Univ. | Universities | Editions |
|---|---|---|---|---|
| 1 | Wood, Wollenberg (& Sheblé), *Power Generation, Operation and Control* | 11 | TAMU, UIUC, GT, UT Austin, UW; ULB, Comillas; FIUBA, UNR, U Chile, USP | 3rd ed. Wiley 2013/14 (TAMU, GT, UT, UW, U Chile, USP); 2nd ed. 1996 (UIUC, ULB, FIUBA). English only. |
| 2 | Gómez-Expósito, Conejo & Cañizares | 8 | (see section 8) | |
| 3 | Baldick, *Applied Optimization* | 2 | UT Austin, UW | |

One university each: Monticelli, *State Estimation* (UIUC); Abur & Gómez-Expósito (GT); Debs (ULB); Billinton & Allan (U Chile); Taylor, *Convex Optimization of Power Systems* (UT); Sioshansi & Conejo, *Optimization in Engineering* (U Chile); Kirchmayer (USP).

Many graduate operation courses use notes or papers only: ETH, DTU, ULiège, UNSJ, and partly GT and UIUC.


### 10. Computational methods for power systems (graduate)
#### Syllabi consulted
- **Texas A&M ECEN 615 Methods of Electric Power Systems Analysis, Fall 2019 (Overbye)**
  URL: https://overbye.engr.tamu.edu/wp-content/uploads/sites/146/2019/08/ECEN_615_Fall2019_Syllabus.pdf
  - Scope: graduate, 43 h. Power flow; sparse matrices; sensitivity analysis and equivalents; data analytics and visualization; OPF and markets; state estimation; high-impact low-frequency events.
  - Prerequisites: ECEN 460.
  - Goals: n.s.
  - Books: Wood, Wollenberg & Sheblé, 3rd ed. 2013 (ISBN 978-0471790556).
  - The Fall 2022 Lecture 1 calls it "the almost required book": https://overbye.engr.tamu.edu/wp-content/uploads/sites/146/2022/09/Lecture-1.pdf
  - The Fall 2023 offering (Birchfield) lists no textbook (only Tinney & Hart's Newton power-flow paper): https://birchfield.engr.tamu.edu/615f23
- **UIUC ECE 530 Analysis Techniques for Large-Scale Electrical Systems (catalog)**
  URL: https://ece.illinois.edu/academics/courses/ece530
  - Scope: graduate, 4 credit hours. Nonlinear systems and power flow; sparsity, storage, visualization; large algebraic systems, parallelization, decomposition; parameter estimation; DAE dynamics.
  - Prerequisites: ECE 464 and ECE 476.
  - Goals: n.s. beyond the description.
  - Books: lecture notes by the instructor.
- **UT Austin EE 394V (Zhu)**, **GT ECE 6320**, **KU Leuven H04C6A** and **ETH 227-0526** are described in sections 8–9.
- **UNICAMP ET931 Análise Computacional de SEE I** (Newton and decoupled load flow, sparse storage): Stevenson; Elgerd (PT); Monticelli, *Fluxo de Carga* (PT); Robba (PT).
- **UNSJ-IEE Flujo de Potencia y Métodos de Optimización**: no books.
- UW-Madison ("ECE 427?"): no syllabus reached.

#### Book ranking
- Wood & Wollenberg: 3 (TAMU, UT Austin, GT).
- Gómez-Expósito, Conejo & Cañizares: 2 (UT Austin, KU Leuven).
- One each: Bergen & Vittal (UT); Baldick (UT); Taylor (UT); Monticelli, *Fluxo de Carga* (UNICAMP); Stagg & El-Abiad (classic, cited in LatAm undergraduate courses).
- The most common "book" is lecture notes: UIUC 530, GT 6320, ETH, DTU, UNSJ.
- No syllabus cites a dedicated PTDF/LODF or sparse-matrix text.


### 11. Electricity markets and power system economics
#### Syllabi consulted
- **UW EE 553 Power System Economics, 2024 (Kirschen)**
  URL: https://peden.ece.uw.edu/academic-ops/wp-content/uploads/sites/5/2024/06/EE553_Syllabus.pdf
  - Scope: graduate, flipped class.
  - Prerequisites: EE 454.
  - Goals: market types; bidding with perfect and imperfect competition; compute LMPs; reliability resources; investment; formulate problems as optimization.
  - Books: R Kirschen & Strbac, *Fundamentals of Power System Economics*, 2nd ed., Wiley 2018. Supplemental: Wood & Wollenberg 3rd; Varian, *Intermediate Microeconomics*.
  - Winter 2023 (Zhang, https://zhangbaosen.github.io/teaching/EE553): Kirschen & Strbac 2nd; references Glover et al.; Baldick.
- **Iowa State EE/Econ 458 Power System Economics (latest offering Fall 2011)**
  URL: https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/syl458team.htm
  - Scope: microeconomics, generator costs, LP and OPF, integer programming and security-constrained UC, market operation, transmission constraints, risk.
  - Prerequisites and goals: n.s.
  - Books: R Kirschen & Strbac, Wiley 2004.
- **UT Austin EE 394V Restructured Electricity Markets: LMP (Baldick)**
  URL: https://users.ece.utexas.edu/~baldick/classes/394V/EE394V.html
  - Scope: graduate. Pricing with and without transmission constraints, hedging, network models, capacity adequacy.
  - Prerequisites and goals: n.s.
  - Books: no single text. References: Glover, Sarma & Overbye 4th; Wood & Wollenberg 2nd; Baldick 2006.
- **UCLouvain LINMA2415 Quantitative Energy Economics**
  URL: https://uclouvain.be/en-cours-2025-linma2415
  - Scope: 5 credits. Duality, equilibrium, economic dispatch/OPF/UC, reserves, hedging, planning.
  - Prerequisites: LP, KKT, duality.
  - Goals: explain market architecture; write mathematical-programming models of markets.
  - Books: textbook Papavasiliou, *Optimization Models in Electricity Markets*. Support Stoft; Kirschen & Strbac.
- **Edinburgh ELEE11054 Power Systems Engineering 5 / PGEE11016 Power Systems Engineering and Economics (MSc)**
  URLs: http://www.drps.ed.ac.uk/25-26/dpt/cxelee11054.htm and http://www.drps.ed.ac.uk/25-26/dpt/cxpgee11016.htm
  - Scope: 10 ECTS. Power flow, OPF, market basics, LMP, PowerWorld.
  - Prerequisites: ELEE10032.
  - Books: Saadat; Glover & Sarma 2002; Jenkins et al., *Embedded Generation*; Kirschen & Strbac 2004; Stoft 2002.
- **Chalmers ENM066**: see section 9.
- **ETH 227-0731-00L Power Market I**
  URL: https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=192898&semkez=2025W&ansicht=ALLE&lang=en
  - Books: handouts only.
- **ULiège ELEC0018 Energy markets and regulation**
  URL: https://www.programmes.uliege.be/cocoon/20252026/en/cours/ELEC0018-1.html
  - Books: none.
- **KTH EG2220 / EG2050**
  URLs: https://www.kth.se/student/kurser/kurs/EG2220?l=en and https://www.kth.se/student/kurser/kurs/EG2050?l=en
  - Books: Söder & Amelin compendium.
- **U. de Chile EL7018 Mercados Internacionales de la Energía, 2014**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=27970
  - Prerequisites: EL4001.
  - Goals: market organisation; simulation models.
  - Books: Kirschen & Strbac; Hunt; Stoft; Gómez-Expósito (ES); Momoh; Wood & Wollenberg 2nd.
- **U. de Chile EL6025 Planificación**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=4428
  - Books: Shahidehpour, Yamin & Li, *Market Operations*; Weber; Mazer.
- **FIUBA 8517 Economía de la Energía Eléctrica**
  URL: https://cms.fi.uba.ar/uploads/8517_3026ab468d.pdf
  - Books: economics texts only.
- **PUC Chile IEE3372 Mercados Eléctricos**: regulation papers only.
- **UNSJ-IEE Mercados de Energía Eléctrica**: no books.

#### Book ranking
1. Kirschen & Strbac, *Fundamentals of Power System Economics*: 6 universities (UW, Iowa State; UCLouvain, Edinburgh, Chalmers; U Chile). 1st ed. 2004; 2nd ed. 2018.
2. Stoft, *Power System Economics*, 2002: 3 (UCLouvain, Edinburgh, U Chile).
3. Wood & Wollenberg (market chapters): 3 (UW, UT, U Chile).
4. One each: Papavasiliou, Hunt, Shahidehpour-Yamin-Li, Söder & Amelin, Varian.


### 12. Operations research: linear and integer optimization
#### Syllabi consulted
- **MIT 6.251J Introduction to Mathematical Programming, Fall 2009**
  URL: https://ocw.mit.edu/courses/6-251j-introduction-to-mathematical-programming-fall-2009/pages/syllabus/ (books on /pages/readings/)
  - Scope: graduate. Linear-optimization geometry, simplex, duality, networks, interior point, SDP, discrete.
  - Prerequisites: n.s.
  - Goals: structure, geometry and algorithms of linear optimization.
  - Books: R Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Athena 1997.
- **MIT 15.093J/6.255J Optimization Methods, Fall 2009**
  URL: https://ocw.mit.edu/courses/15-093j-optimization-methods-fall-2009/pages/syllabus/
  - Books: R Bertsimas & Tsitsiklis.
- **MIT 15.053 Optimization Methods in Management Science, Spring 2013**
  URL: https://ocw.mit.edu/courses/15-053-optimization-methods-in-management-science-spring-2013/pages/syllabus/
  - Scope: undergraduate. LP, IP, networks, Excel Solver.
  - Prerequisites: none formal.
  - Books: none required. Optional Bradley, Hax & Magnanti, *Applied Mathematical Programming* (1977, free online).
- **UC Berkeley IEOR 162 Linear Programming**
  URL: https://www.ocf.berkeley.edu/~abhardwaj/Teaching/162syllabus.pdf (Spring 2023 offering: https://lavaei.ieor.berkeley.edu/Course_IEOR162_Spring_2023.html)
  - Scope: 3 units.
  - Prerequisites: Math 53/54.
  - Goals: model decision problems; learn solution methods.
  - Books: R Winston & Venkataramanan, *Introduction to Mathematical Programming*, 4th ed. Rec Fourer, Gay & Kernighan, *AMPL*, 2nd ed.
- **Cornell ORIE 3300/5300, Fall 2019**
  URL: https://people.orie.cornell.edu/aslewis/2019%203300%20outline.pdf
  - Scope: LP, simplex, duality, branch and bound, AMPL.
  - Prerequisites: linear algebra.
  - Books: no formal text; the *AMPL* book.
- **Georgia Tech ISyE 6669 Deterministic Optimization**
  URLs: https://www2.isye.gatech.edu/~jsokol/6669/syllabus.pdf (R Winston, *Operations Research*, 3rd ed.; prerequisite ISyE 4231) and https://omscs.gatech.edu/sites/default/files/documents/Syllabi/ISYE%206669%202025-1.pdf (Spring 2025: no textbook; references Rardin, Boyd & Vandenberghe, Ben-Tal & Nemirovski; prerequisites linear algebra, calculus, probability, Python)
- **UIUC IE 411 Optimization of Large-Scale Linear Systems, Fall 2025**
  URL: https://ws.engr.illinois.edu/custom/getsyllabus.asp?id=3239
  - Prerequisites: IE 310; MATH 257/415.
  - Goals: theory and computation of LP.
  - Books: R Bertsimas & Tsitsiklis; R Bazaraa, Jarvis & Sherali, *Linear Programming and Network Flows*, 4th ed.; optional Vanderbei, 4th ed.
- **UW-Madison 525 Linear Optimization (Del Pia)**
  URL: https://sites.google.com/site/albertodelpia/teaching
  - Books: Bertsimas & Tsitsiklis; the integer course uses Conforti, Cornuéjols & Zambelli.
- **DTU 42101 Operations Research**
  URL: https://kurser.dtu.dk/course/42101
  - Scope: BSc, 5 ECTS.
  - Prerequisites: linear algebra.
  - Books: Hillier & Lieberman, latest ed.
- **DTU 42114 Integer Programming**
  URL: https://kurser.dtu.dk/course/42114
  - Prerequisites: 42101.
  - Goals: solve production, energy and transport problems.
  - Books: Wolsey, *Integer Programming*.
- **DTU 42136 Advanced OR (decomposition in Julia/JuMP)**
  URL: https://kurser.dtu.dk/course/42136
  - Books: papers; Desrosiers et al., *Branch-and-Price* (free).
- **KU Leuven HBE13E Operations Research**
  URL: https://onderwijsaanbod.kuleuven.be/syllabi/e/HBE13E
  - Prerequisites: HBE07E.
  - Books: Rec Winston, *Operations Research: Applications and Algorithms*, 4th ed.
- **ETH 401-3901-00L Linear & Combinatorial Optimization**
  URL: https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=193687&semkez=2025W&ansicht=ALLE&lang=en
  - Books: Korte & Vygen; Schrijver (two books); Ahuja, Magnanti & Orlin.
- **KTH SF1841 Optimization / SF2812 Applied Linear Optimization**
  URLs: https://www.kth.se/student/kurser/kurs/SF1841?l=en and https://www.kth.se/student/kurser/kurs/SF2812?l=en
  - Books: Nash & Sofer, *Linear and Nonlinear Programming*.
- **UCLouvain LINMA2491 Operational Research**
  URL: https://uclouvain.be/en-cours-2025-linma2491
  - Books: Conforti, Cornuéjols & Zambelli; Birge & Louveaux; Sun & Conejo, *Robust Optimization in Electric Energy Systems*.

#### Syllabi consulted — Latin America
- **FIUBA 9104 Modelos y Optimización I, 2020**
  URL: https://cms.fi.uba.ar/uploads/9104_fc72e669ac.pdf
  - Scope: LP modelling, simplex, sensitivity, IP/MIP, heuristics intro.
  - Prerequisites: n.s.
  - Books: Winston 4ª ed. 2006 (ES); Hillier & Lieberman 8ª ed. 2008 (ES); Taha 7ª ed. (ES); Williams, *Model Building*; Bazaraa-Jarvis-Sherali; Eppen-Gould.
- **UTN FRBA Investigación Operativa, Plan 23**
  URL: https://frba.utn.edu.ar/wp-content/uploads/2025/03/Investigacion-Operativa_23.pdf
  - Scope: year 4, 96 h.
  - Prerequisites: Probabilidad y Estadística, Análisis Numérico.
  - Books: R Hillier & Lieberman 2015 (ES); Taha (Alfaomega 2012, ES); Winston (Thomson 2005, ES).
- **UTN FRBB Investigación Operativa, 2020–25**
  URL: https://www.frbb.utn.edu.ar/frbb/info/departamentos/loi/programas_analiticos/investigacion_operativa.pdf
  - Books: R Eppen-Gould; Mathur-Solow; Winston (ES). Rec Ahuja-Magnanti-Orlin; Hillier-Lieberman 7ª (ES); Taha 7ª (ES).
- **U. de Chile EL4114 Optimización (Electrical), 2022**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=89206
  - Prerequisites: EL3104.
  - Books: R Bertsimas & Tsitsiklis; Boyd & Vandenberghe; Sioshansi & Conejo 2017. Rec Conejo et al., *Decomposition Techniques*.
- **U. de Chile IN3171 Modelamiento y Optimización**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=235674
  - Books: R Bertsimas & Tsitsiklis; Boyd & Vandenberghe; Conforti-Cornuéjols-Zambelli. Rec Ahuja-Magnanti-Orlin; Nemhauser & Wolsey; Schrijver.
- **U. de Chile MA3701 Optimización**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=185462
  - Books: Boyd & Vandenberghe; Chvátal; Hillier & Lieberman 6ª (ES); Luenberger & Ye 4e; Bazaraa-Jarvis-Sherali 4e; Bazaraa-Sherali-Shetty 3e; Nocedal & Wright 2e.
- **PUC Chile ICS1113 Optimización**
  URL: https://catalogo.uc.cl/index.php?tmpl=component&option=com_catalogo&view=programa&sigla=ICS1113
  - Books: Bazaraa-Jarvis-Sherali; Luenberger; Spanish texts.
- **USP PRO3341**
  URL: https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=PRO3341
  - Books: Winston & Venkataramanan 4e; Hillier & Lieberman, *Introdução à Pesquisa Operacional* (PT 2006); Arenales et al. (PT).
- **UNAM Investigación de Operaciones II**
  URL: https://www.dgire.unam.mx/images/planes/lic/51/1747.pdf
  - Books: R Winston 4a (ES); Hillier & Lieberman 7a (ES); Taha 6a (ES).

#### Book ranking

| # | Book | Univ. | Universities | Editions / languages |
|---|---|---|---|---|
| 1 | Winston, *Operations Research: Applications and Algorithms* / *Introduction to Mathematical Programming* | 7 | Berkeley, GT, KU Leuven; FIUBA, UTN, UNAM, USP | ES *Investigación de operaciones: aplicaciones y algoritmos*, 4ª ed., Thomson 2005 |
| 2 | Hillier & Lieberman, *Introduction to Operations Research* | 6 | DTU; FIUBA, UTN, UNAM, U Chile, USP | ES *Introducción a la investigación de operaciones* 6ª–10ª; PT 2006 |
| 3 | Bertsimas & Tsitsiklis, *Introduction to Linear Optimization*, Athena 1997 | 4 (+CMU optional) | MIT, UIUC, UW-Madison, U Chile | |
| 3 | Bazaraa, Jarvis & Sherali, *Linear Programming and Network Flows* | 4 | UIUC, FIUBA, U Chile, PUC | |
| 5 | Taha, *Operations Research: An Introduction* | 3 | FIUBA, UTN, UNAM | ES *Investigación de operaciones* 6ª–9ª ed. |
| 5 | Ahuja, Magnanti & Orlin, *Network Flows* | 3 | ETH, UTN, U Chile | |
| 5 | Conforti, Cornuéjols & Zambelli, *Integer Programming* | 3 | UW-Madison, UCLouvain, U Chile | |
| 5 | Eppen & Gould | 3 | FIUBA, UTN, UNAM | |
| 5 | Schrijver | 3 | ETH, CMU (optional), U Chile | |
| 10 | Fourer, Gay & Kernighan, *AMPL* | 2 | Berkeley, Cornell | |

One university each: Wolsey (DTU); Nash & Sofer (KTH); Vanderbei (UIUC); Rardin (GT); Bradley, Hax & Magnanti (MIT); Korte & Vygen (ETH); Chvátal (U Chile).


### 13. Convex and nonlinear optimization
#### Syllabi consulted
- **MIT 6.079 Introduction to Convex Optimization, Fall 2009**
  URL: https://ocw.mit.edu/courses/6-079-introduction-to-convex-optimization-fall-2009/pages/syllabus/ (books on /pages/readings/)
  - Scope: convex sets and functions, LS/LP/QP/SDP, duality.
  - Prerequisites: n.s.
  - Goals: recognize and model convex problems.
  - Books: R Boyd & Vandenberghe, *Convex Optimization*, Cambridge UP 2004 (free online). Suggested Bertsekas, *Convex Optimization Theory* 2009; Ben-Tal & Nemirovski 2001.
- **MIT 6.252J Nonlinear Programming, Spring 2003**
  URL: https://ocw.mit.edu/courses/6-252j-nonlinear-programming-spring-2003/pages/syllabus/
  - Scope: graduate. Gradient/Newton/quasi-Newton, interior point, Lagrange multipliers, duality; applications include power systems.
  - Prerequisites: n.s.
  - Goals: unified analytical and computational approach.
  - Books: R Bertsekas, *Nonlinear Programming*, 2nd ed., Athena 1999.
- **Stanford EE364a Convex Optimization I**
  URL: https://web.stanford.edu/class/ee364a/
  - Books: Boyd & Vandenberghe. Prerequisites and goals n.s. on page.
- **Stanford CME307/MS&E311 Optimization**
  URL: https://web.stanford.edu/class/msande311/
  - Prerequisites: Math 113, 115.
  - Books: Luenberger & Ye, *Linear and Nonlinear Programming*, 5th ed.
- **CMU 10-725 Convex Optimization, Fall 2019**
  URL: https://www.stat.cmu.edu/~ryantibs/convexopt/syllabus.pdf
  - Prerequisites: algorithms, mathematical maturity.
  - Books: lectures self-contained; references Boyd & Vandenberghe, Rockafellar.
- **CMU 10-725, Fall 2012**
  URL: https://www.cs.cmu.edu/~ggordon/10725-F12/index.html
  - Books: Boyd & Vandenberghe. Optional Bertsimas & Tsitsiklis, Bertsekas *Nonlinear Programming*, Rockafellar, Schrijver.
- **GT ISyE 6669 (Spring 2025)**: see section 12.
- **KTH SF2822 Applied Nonlinear Optimization**
  URL: https://www.kth.se/student/kurser/kurs/SF2822?l=en
  - Books: Nash & Sofer.
- **UT Austin EE 394V (Zhu)**: Boyd & Vandenberghe, Baldick.
- **U. de Chile EL4114 / IN3171 / MA3701**: Boyd & Vandenberghe (all 3). MA3701 adds Nocedal & Wright 2e, Luenberger & Ye, Bazaraa-Sherali-Shetty.
- **PUC Chile ICS1113**: Luenberger.
- DTU 02612 Constrained Optimization: no literature field.

#### Book ranking
1. Boyd & Vandenberghe, *Convex Optimization*, 2004: 6 universities (MIT, Stanford, CMU, GT, UT Austin; U Chile).
2. Luenberger (& Ye), *Linear and Nonlinear Programming*: 3 (Stanford, U Chile, PUC).
3. Two each: Bertsekas, *Nonlinear Programming* (MIT, CMU optional); Ben-Tal & Nemirovski (MIT, GT); Baldick (UT, UW).
4. One each: Nocedal & Wright (U Chile); Nash & Sofer (KTH); Bazaraa-Sherali-Shetty (U Chile); Rockafellar (CMU).


### 14. Evolutionary computation and metaheuristics
#### Syllabi consulted
- **VU Amsterdam X_400111 Evolutionary Computing**
  URL: https://studiegids.vu.nl/en/courses/2026-2027/X_400111
  - Scope: MSc, 6 EC.
  - Prerequisites: Python (strict).
  - Goals: evolutionary methods as solvers and simulators; design choices.
  - Books: Eiben & Smith, *Introduction to Evolutionary Computing*, 2nd ed. 2015; 2026-27 adds Eiben, Miras & Hart, *Robot Evolution*.
- **Marquette COEN 4870/5870**
  URL: http://povinelli.eece.mu.edu/teaching/coen4870/syllabus.html
  - Prerequisites: data structures, Calculus 1, discrete math.
  - Goals: implement and compare algorithms.
  - Books: R Eiben & Smith, 2nd ed.
- **Univ. of Memphis COMP 7282/8282, Spring 2020**
  URL: https://www.memphis.edu/cs/courses/syllabi/7282.pdf
  - Prerequisites: COMP 6601.
  - Books: suggested Eiben & Smith; Simon, *Evolutionary Optimization Algorithms* 2013; Dasgupta & Michalewicz. References Goldberg 1989; *Handbook of EC*; Yang.
- **Michigan State CSE 848, Fall 2020**
  URL: https://cse.msu.edu/~cse848/syllabus2020.html
  - Prerequisites: graduate standing.
  - Books: none required. Rec Banzhaf; Deb; Eiben & Smith; Goldberg; Mitchell; Poli et al.; Simon; Burke & Kendall.
- **Univ. of Oklahoma CS 5970, Fall 2010**
  URL: http://www.cs.ou.edu/~hougen/classes/Fall-2010/EC/materials/Syllabus.html
  - Prerequisites: CS 2413.
  - Books: R De Jong, *Evolutionary Computation: A Unified Approach*, 2006.
- **Universidad de Granada Metaheurísticas, 2026/27**
  URL: https://www.ugr.es/estudiantes/grados/grado-ingenieria-informatica/metaheuristicas-ecomputacy-sistinteligentes/guia-docente
  - Scope: year 3, 6 ECTS.
  - Books: Alba; Pardalos & Resende; Dorigo & Stützle; Eiben & Smith 2nd; Du & Swamy; Chopard & Tomassini. Complementary Talbi 2009.
- **DTU 42137 Optimization using Metaheuristics**
  URL: https://kurser.dtu.dk/course/42137
  - Prerequisites: 42101.
  - Books: *Handbook of Metaheuristics*; *Search Methodologies*; *Stochastic Local Search*.
- **Chalmers FFR105 Stochastic Optimization Algorithms**
  URL: https://www.chalmers.se/en/education/your-studies/find-course-and-programme-syllabi/course-syllabus/FFR105/?acYear=2025/2026
  - Books: Wahde, *Biologically Inspired Optimization Methods*.
- **Eastern Michigan EC**
  URL: https://emunix.emich.edu/~mevett/ECCourse/syllabus.html
  - Books: Langdon & Poli; Koza.
- **Leiden Evolutionary Algorithms**: UNVERIFIED (page now 404; search snippet only).
- **USP PEA3422** (see section 9): Goldberg; Michalewicz 3e; Deb 2001.
- **FIUBA 7120 Modelos y Optimización III**
  URL: https://cms.fi.uba.ar/uploads/7120_1c0a8b2473.pdf
  - Scope: tabu search, simulated annealing, genetic algorithms, GRASP.
  - Books: A. Díaz, *Optimización heurística y redes neuronales* (ES); Taha; Hillier & Lieberman.
- **U. de Chile IN7605 Heurísticas**
  URL: https://ucampus.uchile.cl/m/fcfm_catalogo/programa?bajar=1&id=210934
  - Books: Berthold-Lodi-Salvagnin; Taillard.
- No syllabus covers quality-diversity or MAP-Elites.

#### Book ranking
1. Eiben & Smith: 5 (VU, Marquette, Memphis, MSU, Granada; +Leiden unverified). 1st ed. 2003; 2nd ed. Springer 2015.
2. Goldberg, *Genetic Algorithms in Search, Optimization and Machine Learning*, 1989: 3 (Memphis, MSU, USP).
3. Two each: Simon 2013 (Memphis, MSU); Deb, *Multi-Objective Optimization Using EAs* (MSU, USP); Burke & Kendall, *Search Methodologies* (MSU, DTU).
4. One each: De Jong; Michalewicz; Talbi; Wahde; Dorigo & Stützle; Díaz (ES).


### 15. Scientific computing, HPC and GPU
#### Syllabi consulted
- **MIT 6.172 Performance Engineering, Fall 2018**
  URL: https://ocw.mit.edu/courses/6-172-performance-engineering-of-software-systems-fall-2018/pages/syllabus/
  - Prerequisites: 6.004, 6.006, 6.031.
  - Goals: build scalable high-performance software.
  - Books: none ("reading materials posted").
- **UIUC ECE 408/CS 483 Applied Parallel Programming, Summer 2025**
  URL: https://lumetta.web.engr.illinois.edu/408-Sum25/slide-copies/ece408-lecture1-introduction-Sum25-x4.pdf
  - Prerequisites: ECE 220 or CS 225.
  - Goals: program massively parallel (GPU) processors.
  - Books: Hwu, Kirk & El Hajj, *Programming Massively Parallel Processors*, 4th ed. 2022.
- **UC Berkeley CS C267, Spring 2023**
  URL: http://msse.berkeley.edu/wp-content/uploads/2022/09/S23-COMPSCI-C267-Syllabus.pdf
  - Scope: OpenMP, GPU, MPI, parallel linear algebra.
  - Books: none.
- **Georgia Tech CSE 6220**
  URL: https://cse6220.gatech.edu/sp24-oms/
  - Prerequisites: CS 3510, C/C++.
  - Books: Grama, Gupta, Karypis & Kumar, *Introduction to Parallel Computing*, 2003. The 2026 syllabus lists no book.
- **Cornell CS 5220, Spring 2026**
  URL: https://www.cs.cornell.edu/courses/cs5220/2026sp/syllabus.html
  - Prerequisites: C++, CS 3410.
  - Books: none required. Rec Hager & Wellein; Kirk & Hwu; Pacheco; McCool et al.
- **UIUC CS 450 Numerical Analysis**
  URL: https://siebelschool.illinois.edu/academics/courses/cs450
  - Books: Heath, *Scientific Computing: An Introductory Survey*, 2nd ed.
- **Cornell CS 4220, Spring 2024**
  URL: https://www.cs.cornell.edu/courses/cs4220/2024sp/
  - Books: Ascher & Greif; Trefethen & Bau; Demmel.
- **DTU 02601**
  URL: https://kurser.dtu.dk/course/02601
  - Books: Cheney & Kincaid 7th.
- **DTU 02614 HPC**
  URL: https://kurser.dtu.dk/course/02614
  - Books: notes.
- **ETH 401-0663-00L Numerical Methods for CS**
  URL: https://www.vorlesungen.ethz.ch/Vorlesungsverzeichnis/lerneinheit.view?lerneinheitId=192908&semkez=2025W&ansicht=ALLE&lang=en
  - Books: Dahmen & Reusken; Gander, Gander & Kwok; Deuflhard & Hohmann.
- **KTH DD2356**
  URL: https://www.kth.se/student/kurser/kurs/DD2356?l=en
  - Books: Patterson & Hennessy.
- **KTH DD2360 Applied GPU Programming**
  URL: https://www.kth.se/student/kurser/kurs/DD2360?l=en
  - Books: not public.
- **USP MAP2210**
  URL: https://uspdigital.usp.br/jupiterweb/obterDisciplina?sgldis=MAP2210
  - Books: Noble; Strang; Demmel; Burden & Faires. Scope not extracted.
- No Latin American HPC/GPU course was found. No syllabus covers JAX.

#### Book ranking
1. Kirk & Hwu: 2 (UIUC, Cornell).
2. Demmel, *Applied Numerical Linear Algebra*: 2 (Cornell, USP).
3. Everything else: 1 each (Grama; Hager & Wellein; Heath; Ascher & Greif; Cheney & Kincaid; Gander et al.; Patterson & Hennessy).
- Four major HPC courses use no textbook (MIT, Berkeley, Cornell, DTU).
