#import "@preview/polylux:0.3.1": *
#import themes.university: *
#import "@preview/gentle-clues:1.1.0": *
#set math.equation(numbering: "(1)")

#show: university-theme.with(
  color-a: rgb("#1A3C6E"),
  color-b: rgb("#1A3C6E"),
  short-author: "Noe Lomidze",
  short-title: "Final Project",
  short-date: "Fall Term 2024",
)

#title-slide(
  authors: ("Noe Lomidze"),
  title: "Final Project",
  subtitle: "Sturm-Liouville problem",
  date: "January 9, 2025",
  institution-name: "Kutaisi International University",
  logo: align(left, image("Images/thumbnail_KIU2-1.png", width: 30%)),
)

#pagebreak()

#slide(outline(fill: line(length: 100%)))

#slide(title: "Tasks")[
  #info(title: [Description])[
    - Formulate algorithm, explain your approach in written.
    - Describe properties of numerical methods written.
    - Develop test cases and demonstrate validity of your results.
    - Upload all necessary files, including
      + Presentation file
      + Code
      + Test data and their description
    - Using shooting method and ball motion equation is compulsory
  ]
]

/* title: none,
 * header: none,
 * footer: none,
 * new-section: none, */

#slide(title: "Sturm-Liouville problem")[
  #info(title: "Components")[
    - #text(fill: eastern, "Input"): #text(fill: blue, "Sturm-Liouville problem")
  - #text(fill: eastern, "Task"): find first 8 eigenvalues and eigenfunctions
- #text(fill: eastern, "Approach"): approximate vanishing or singular coefficients
- #text(fill: eastern, "Output"): visualisation of eigenvalues and eigenfaunctions
- #text(fill: green, "Test"): test case description
- #text(fill: maroon, "Methodology"): should contain problem formulation, including equation with initial and boundary condition, method of solution, algorithm

]
]

#slide(title: "Examples, Sturm-Liouville Problem")[
  #abstract(title: [Theory])[
    #text(size: 0.8cm, "In mathematics and its applications, a ")
    *Sturm-Liouville*
    #text(size: 0.8cm, "problem is a second-order linear ordinary differential equation of the form:")
    $
    d/ (dif x )[p(x) (dif y) /(dif x)]
    + q(x) y = - lambda w(x) y
    $ <thing>
    for the given functions $p(x),q(x) "and" w(x)$ together with some #text(fill: blue, )[
      #link("https://en.wikipedia.org/wiki/Boundary_value_problem")[Boundary Conditions]
    ] at extreme values of $x$. The goals are:
    - #text(size: 0.6cm, "To find the λ (eigenvalue) for which there exists a non-trivial solution to the problem.")
  - #text(size: 0.6cm, "To find the corresponding solution") $script(y=y(x))$ #text(size: 0.6cm, "of the problem, such functions are eigenfunctions")
]
]

#slide(title: "Main results")[
  The main results in Sturm-Liouville theory apply
  to a Sturm-Liouville problem: #h(0.3cm) $d/ (dif x )[p(x) (dif y) /(dif x)]
  + q(x) y = - lambda w(x) y$ <label-eq> \
  #text(size: 23pt, "on a finite interval") $script([a, b])$
  #text(size: 23pt, "that is \"regular\". The problem is regular if: ")
  #text(size: 23pt)[
    - the coefficient functions $p, q, w$ and derivative $p'$ are all continuous on $[a, b]$;
    - $p(x) > 0$ and $w(x) > 0$ for all $x in [a, b];$
    - the problem has #text(fill: blue)[
      #link("https://en.wikipedia.org/wiki/Robin_boundary_condition")[separated boundary conditions]] of the form
  ]
  #text(size: 23pt)[
    $
    alpha_1y(a) + alpha_2y'(a) = 0 space space space
    alpha_1, alpha_2 "not both 0,"
    $
    $
    beta_1y(b) + beta_2y'(b) = 0 space space space
    beta_1, beta_2 "not both 0,"
    $
    #text(size: 22pt)[
      The function $w = w(x)$ is called the _weight_ or _density_ function.
    ]
  ]
]

#slide(title: "Reduction to Sturm–Liouville form")[
  The differential @thing is said to be in
  *Sturm-Liouville* or \ *self-adjoint form* \
  #text(fill: blue, size: 31pt)[
    #link("https://en.wikipedia.org/wiki/Bessel_function")[*Bessel equation*]
  ]
  \
  #v(0.1pt)
  #h(1.5cm)$x^2y'' + x y' + (x^2 - nu^2)y = 0$
  \

  #text(size: 27pt)[
    which can be written in Sturm-Liouville form (first by dividing through by x, then by collapsing the first two terms on the left into one term) as:
  ]
  #v(0.1pt)
  #h(1.3cm)
  #text(size: 30pt)[
    $(x y')' + (x - nu^2/x) y = 0$
  ]
]

#slide(title: "Simple example")[
  #text(size: 20pt)[
    $"For" lambda in RR, " solve:        "$ #h(20pt)
    $y'' + lambda y = 0$ $"  "y(0) = 0, "  "y'(pi)= 0$ <SimpleExample>
  ]
  \
  #text(size: 18pt)[
    *Case 1.* Let $lambda < 0.$ Then $lambda = - mu^2, space mu in RR \\{0}.$ Solution of ODE is
    #text(fill: rgb("#e34646"))[$y(x) = A e^(mu x) + B e^(-mu x)$]
    \
    This $y$ satisfies boundary conditions iff $A = B = 0 ==> y eq.triple 0$. So there are no negative eigenvalues..
    *Case 2.* Let $lambda = 0.$ In this case, it easily follows that trivial solution is the only solution of
    #align(center)[$y''=0,space space y(0) = 0, space space
      y'(pi) = 0. "Thus," 0" is not an eigenvalue. "
    $]
    *Case 3.* Let $lambda > 0$. Then $lambda = mu^2$, where $mu in R \\{0}$. The general solution of ODE is given by
    #align(center)[#text(fill: rgb("#e34646"))[$y(x) = A cos(mu
    x) + B sin(mu x)$]]

    We need:
    #text(fill: rgb("#2c9228"))[$A = 0 "and" B cos(mu pi) = 0$.]
    But $B cos(mu pi) = 0$ iff either $B = 0 "or" cos(mu pi) = 0.$\

    If $A = 0 "and" B = 0 => y eq.triple 0,$
    Thus $cos(mu pi) = 0 "should hold"$, the last equation has solutions given by $mu = (1/2 + n)$, for $n = 0, plus.minus 1, plus.minus 2, dots$ Thus the eigenvalues are
    #text(fill: blue)[$lambda_n = (1/2 + n)^2, space$ ]
    $n = 0,1, 2, dots$
    \
    and corresponding eigenfunctions are given by
    $phi.alt_n = B sin((1/2 + n) x) $
    \
    _Note_: #text(fill: rgb("#e37070"))[
      All the eigenvalues are positive. The eigenfunctions corresponding to each eigenvalue form
    a one dimensional vector space and so the eigenfunctions are unique upto a constant multiple.]
  ]
]

#slide(title: [Regular SL-BVP properties])[
  #text(size: 20pt)[
    Eigenvalues of regular _SL-BVP_ are real.
    #text(fill: rgb("#27927d"))[
      $cal(L)[y] eq.triple d/ (dif x)[p(x) (dif y)/(dif x)] + q(x)y, space$ #text(fill: blue)[$ cal(L)[y]+lambda r(x)y = 0$]
    ]
    #idea(title: [Proof])[
      #text(size: 18pt)[
        Suppose $lambda in CC$ is an eigenvalue and $y$ be the corresponding eigenfunction. That is,
        \
        $cal(L)[y] + lambda r(x) y = 0, space
        a_1y(a) + a_2p(a)y'(a)= 0, space
        b_1y(b) + a_2p(b)y'(b)= 0$, Taking compl conj
        \
        $cal(L)[overline(y)] + lambda r(x) overline(y) = 0, space
        a_1overline(y)(a) + a_2p(a)overline(y)'(a)= 0, space
        b_1overline(y)(b) + a_2p(b)overline(y)'(b)= 0$
        \
        Multiply the first ODE with $overline(y)$ and multiply that with $y$, subtracting one from another yields:
        \
        #text(fill: blue)[
          $"          "
          [p(y'overline(y) - overline(y)'y)]' +
          (lambda - overline(lambda))r y overline(y) = 0$
        ], #h(50pt) Integrating the last equation yields:

        $"                               "[p(y'overline(y) - overline(y)'y)] bar_a^b = -(lambda -
        overline(lambda)) integral_a^b r(x) |y(x)|^2 dif x.$

        But LHS is zero, since we have both boundary conditions, also we know that $b_1^2 + b_2^2 != 0" Thus"$
        $"                                             "(lambda - overline(lambda)) integral_a^b r |y|^2 dif y = 0$
        \
        Since $y$ being an eigenfaunction $y != 0, "also" r > 0,
        "only possibility is that" lambda = overline(lambda) "which  means that" lambda "is real.                                                                       Done."$
      ]
    ]
  ]

  #text(size: 20pt)[
    // The _eigenfunctions_ of a regular _SL-BVP_ corresponding
    // to distinct _eigenvalues_ are orthogonal _w.r.t_ 
    // weight function $r "on" [a, b],$ that is, 
    Eigenfunctions of the distinct eigenvalues, of a regular SL-BVP  are
othogonal:
    \
    #text(size: 20pt)[
      $ integral_a^b r(x) u(x) v(x) = 0 $
    ]
    #idea(title: [Proof])[
      As in the previous proof, writing down the equations satisfied by $u "and" v$, and multiplying the equation for $u$ with $v$ and vice versa, finally substracting we get:
      #text(fill: blue)[$"                               "[p(u'v - v'u)] + (lambda - mu) r u v = 0$]

      Integrating the last equality yields: \
      #text(fill: blue)[
        $"                     "[p(u'v - v'u)] |_a^b = -(lambda - mu)
        integral_a^b r(x) u(x) v(x) dif x$
      ]

      Reasoning exactly as in the previous proof, LHS is zero, since $lambda != mu$, proof is done.
    ]
  ]
]


#let eulerCode = ```python
    def euler_method(f, t0, y0, h, t_end):
        t_values = [t0]
        y_values = [y0]
        while t_values[-1] < t_end:
            t_new = t_values[-1] + h
            y_new = y_values[-1] + 
            h * f(t_values[-1], y_values[-1])
            t_values.append(t_new)
            y_values.append(y_new)

        return np.array(t_values), np.array(y_values)```

#let rungeCode = ```python
def rk4_method(f, t0, y0, h, t_end):
    t_values = [t0]
    y_values = [y0]
    while t_values[-1] < t_end:
        t = t_values[-1]
        y = y_values[-1]
        k1 = h * f(t, y)
        k2 = h * f(t + h/2, y + k1/2)
        k3 = h * f(t + h/2, y + k2/2)
        k4 = h * f(t + h, y + k3)
        y_new = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)
        t_new = t + h
        t_values.append(t_new)
        y_values.append(y_new)
    return np.array(t_values), np.array(y_values)
```

#slide(title: [Numerical Methods for ODEs])[
  There are many methods to solve $d /(dif y) = f(t, bold(y))$, but lets consider two:

  _Euler's method_: #text(fill: blue)[$y_(j + 1) = y_j + k f_j, space space"which is" O(k)$]
  \
  #code()[
    #text(size: 15pt)[
      #eulerCode
    ]
  ]

  // _Runge-Kutta method_: #text(fill: blue)[
  //   $$
  // ]
  
  _Classical Runge-Kutta Method_: 
  #text(fill: blue)[
    $ y_(j + 1) = y_j + 1/6 (k_1 + 2k_2+2k_3+k_4), space
    space space O(k^4) $
  
  ]
  _Where_:
  \
  #align(left)[
    #align(left)[
    #text(size: 30pt, fill: blue)[
    $k_1 = k f_j$ \ 
    $k_2 = k f(t_j + k/2, y_j + 1/2 k_1)$\
    $k_3 = k f(t_j + k/2, y_j + 1/2k_2)$\
    $k_4 = k f(t_(j + 1), y_j + k_3)$]
    ]

  ]
  \
  #code[
      #text(size: 17pt)[
        #rungeCode
      ]
  ]
]


#slide(title: [Shooting method for BVPs])[
  #text(size: 20pt)[
    In numerical analysis, the 
    #link("https://en.wikipedia.org/wiki/Shooting_method")[*shooting method*]
     is a method
    for solving a boundary value problem by recuding it to
    an #link("https://en.wikipedia.org/wiki/Initial_value_problem")[#text(fill: blue)[initial value problem.]]
    \
    Example: $
               w''(t) = 3/2 w^2(t), space space 
               w(0) = 4, space space w(1) = 1, "to the initial value problem"
             $
             $
               w''(t) = 3/2 w^2(t), space space
               w(0) = 4, space space w'(0) = s
             $
    

  ]
  #align(center)[After solving using different methods for $s$, we get ]
  $
  w'(0)= -8 "and" w'(0) = -35.9 "(approximately)"
  $
  \
  #figure(
    
    image("Images/Shooting_method_trajectories.svg.png", width: 15cm, height: 10cm),
    caption: [Trajectories $w(t; s) "for" s=w'(0) "equal to"$ $-7, -8, -10, -36 "and" -40$ ]
    
  )
  #figure(
    image("Images/Shooting_method_error.svg.png", width: 15cm, height: 10cm),
    caption: [The function $F(s) = w(1; s) - 1$]
  )
  
    
  
]

#slide(title: [Previous example solved: @SimpleExample[Problem]])[
  #text(size: 20pt)[
    By hand we got $lambda_n = (1/2 + n)^2, space 
    phi.alt_n (x) = sin((1/2 + n)x)$ \
    In python file: `SimpleSL_example.py` I used
    *shooting method* with *RK4* to do the same as before,
    with bisection method, I was able to find the eigenvalues iteratively 
    taking midpoints of interval $[lambda_min, lambda_max]$, checking the sign of $y'(pi)$ and so on..
  ]
  
  
  #code()[
    ```py
    def runge_kutta_4(f, y0, t, h):
    def shooting_method(lambda_val, x, h):
    def find_eigenvalues(n_eigenvalues, x_points):
    def compute_eigenfunction(lambda_val, x, h):
    ```
  ]
  
  
  #figure(
    image("Images/SimpleExample.png"),
  )

  #align(center)[Eigenvalues vs Analytical Values:
    #image("Images/Result.png", width: 17cm)
  ]

  // #table(
  // columns: 4,
  // table.header(
  //   [$n$], [Numerical],
  //   [Analytical], [Error (%)],
  // ),
  
  // [1], [0.250000], [0.250000], [0.000048],
  // [2], [2.250000], [2.250000], [0.000001],
  // [3], [6.250001], [6.250000], [0.000016],
  // [4], [12.250000], [12.250000], [0.000002],
  // [5], [20.250011], [20.250000], [0.000055],
  // [6], [30.250031], [30.250000], [0.000103],
  // [7], [42.250075], [42.250000], [0.000176],
  // [8], [56.250180], [56.250000], [0.000319],
  // )
  

]

#slide(title: [Another simple example])[
  Almost the same equation, but different boundary points: \
  $ ""y'' + lambda y = 0, y(0) = 0, y(1) = 0 $
  \
  Solution by inspection is: 
  $
  A cos(sqrt(lambda) x) + B sin(sqrt(lambda)x)
  $
  
  But because of boundary conditions we get:  $
  sin(sqrt(lambda)x) = 0 "so"$
  $
    lambda_n = pi^2n^2, space y_n (x) = sin(pi n x)
  $
  #figure(
    image("Images/easierResult.png", 
    width: 18cm, height: 13cm),
    
  )
]
#slide(title: [Orthogonality])[
  #idea(title: [Proof])[
    $1.integral_0^1 sin(m pi x) * sin(n pi x) * 1 dif x = 0 space "if" m != n$
    \
    $2.sin(A)sin(B) = 1/2 [cos(A - B) - cos(A + B)]
    "applying it" => $
    \
    $3. (1/2) integral_0^1 [cos((m - n)pi x) - cos((m + n) pi x)]=\
    space space 
     "            "=(1/2) [(sin((m-n)π x)) / ((m-n)π) - 
    (sin((m+n)π x)) / ((m+n)π) ]_0^1 
    $
    
    #text()[
      Since $m$ and $n$ are integers and $m != n ==> $
      $
 
      sin((m - n)pi)= sin((m + n)pi) = 0, "and we're done."
      $  
      
    ]
  ]
]

#slide(title: [Nice example:])[
  #text(size: 23pt)[
    $
      y'' + 3y' + 2' + lambda y = 0, space 
      y(0) = 0, space y(1) = 0.
    $ <Nice_example>
    *Solution* The characteristic equation of that is: 
    $ 
      r^2 + 3r + 2 + lambda, space
      "with zeros" r_(1,2) = (-3plus.minus sqrt(1-4lambda))/2
    $
    *Case 1:* If $lambda < 1/4$ then $r_1 "and" r_2$
    are real and distinct, so the general solution is
    $
      y = c_1e^(r_1t) + c_2e^(r_2t)
    $
    The boundary conditions require that: $c_1 + c_2 = 0 space and space c_1e^(r_1) + c_2e^(r_2) = 0$

    Since the determinant of this system is 
    $e^(r_2) - e^(r_1) != 0,$ the system has only the trivial solution. Therefore $lambda$ is not an eigenvalue of @Nice_example[#text(fill: red)[equation]]
    
  ]
  #text(size: 23pt)[
    *Case 2: *If $lambda = 1/4$ then $r_1 = r_2 = -3/2$ so the general solution of @Nice_example[#text(fill: red)[equation]] is
    $
      y = e^((-3x)/2)(c_1 + c_2x)
    $
    The boundary condition $y(0) = 0$ requires that $c_1 = 0,$ so $y = c_2x e^((-3x)/2)$
    and the boundary condition $y(1) = 0
    $ requires that $c_2$ = 0. Therefore $lambda = 1/4$ is not an eigenvalue.\ 
    *Case 3:* If $lambda > 1/4$ then: 
    $
      r_(1, 2) = -3/2 plus.minus i w space space "with"
    $
    $
      w = sqrt(4 lambda - 1)/2 "or equivalently,"
      space space lambda = (1 + 4w^2)/4
    $
  
    *Case 3(Continued):* In this case the general solution of @Nice_example[#text(fill: red)[equation]] is 
    $
      y = e^((-3x)/2) (c_1cos w x + c_2 sin w x)
    $
    Boundary condition $y(0) = 0$ requires that $c_1 = 0,$ so $y = c_2 e^((-3x)/2)sin w x$
    which holds with $c_2 != 0$ iff $w = n pi$ where $n$ is an integer.\ 
    So the eigenvalues are 
    $
      lambda_n = (1 + 4n^2pi^2)/4,
    $ 
    with associated eigenfunctions

    $
      y_n = e^((-3x)/2) sin n pi x, space space n = 1,2,3, .dots
    $
    
  
  ]
  
  #figure(
    image("Images/NiceExample1.png"),
  )
  

]

#slide(title: [Numerical Solution])[
  #text(size: 23pt)[
    Consider the Sturm-Liouville problem:
    $
      y'' + 3y' + 2y + lambda y = 0, space
      y(0) = 0, space y(1) = 0
    $

    *Shooting Method Approach:*
    1. Convert to first-order system:
    $
      vec(y_1 ', y_2 ') = vec(y_2, -3y_2 - 2y_1 - lambda y_1)
    $
    where $y_1 = y$ and $y_2 = y'$

    2. For each $lambda$, solve IVP with initial conditions:
    $
      y_1(0) = 0, space y_2(0) = 1
    $
    using RK4 method with step size h:
    $
      k_1 = f(t_n, y_n) \
      k_2 = f(t_n + h/2, y_n + (h/2)k_1) \
      k_3 = f(t_n + h/2, y_n + (h/2)k_2) \
      k_4 = f(t_n + h, y_n + h k_3) \
      y_(n+1) = y_n + (h/6)(k_1 + 2k_2 + 2k_3 + k_4)
    $

    3. Define shooting function:
    $
      F(lambda) = y_1(1)
    $
    Eigenvalues occur when $F(lambda) = 0$

    4. Find eigenvalues using bisection:
    For interval $[lambda_L, lambda_R]$, if $F(lambda_L)F(lambda_M) < 0$
    where $lambda_M = (lambda_L + lambda_R)/2$, then eigenvalue exists in $[lambda_L, lambda_M]$

    5. Initial guesses based on analytical solution:
    $
      lambda_n approx (1 + 4n^2 pi^2)/4
    $

    
  ]
]

#slide(title: [Another Good Example])[
  #align(center)[
    #text(35pt, fill: blue)[
      Legendre equation
    ]

    $
      -(1-x^2)y'' + 2x y' + lambda y = 0 "on "[-1,1]
    $
    with boundary conditions $y(-1) = y(1) = 0$

    $
      d/(dif x) [(1 - x^2) (dif y)/(dif x)] + l(l + 1)y = 0
    $

    
  ]
  The above form is a special case of the so-called #highlight()[associated Legendre differential equation] corresponding to the case $m=0$. The Legendre differential equation has regular singular points at $ -1, 1, "and" infinity$
  
  
]
#slide(title: [Another Good Example])[
  #text(25pt)[
    To solve this using the shooting method, we first transform the second-order ODE into a system of first-order ODEs:
    \
  ]
  Let $v = y'$, then:
$
y' &= v \
v' &= (2x v + lambda y)/(1-x^2)
$
This gives us the system:
$
(d)/(dif x) [y] = [v] #h(50pt)
(d)/(dif x) [v] = [(2x v + lambda y)/(1-x^2)]
$


#text(20pt)[
  The shooting method converts our boundary value problem into an initial value problem:

- At $x = -1$:

  - We know $y(-1) = 0$ (given boundary condition)
  - We guess $y'(-1) = 1$ (arbitrary non-zero value)

For a given eigenvalue guess $lambda$:

  1. Set initial conditions $y(-1) = 0$, $y'(-1) = 1$
  2. Integrate the system from $x = -1$ to $x = 1$
  3. Check the value of $y(1)$

Eigenvalue Search,
Define the shooting function:
$F(lambda) = y(1)$
The eigenvalues are the values of $lambda$ where $F(lambda) = 0$, then runge kutta, and then bisection method for finding eigenvalues.
]
]

#let lastRunge = ```python
def runge_kutta_4(f, x, y, h):```

#let forwEuler = ```python
def forward_euler(f, x, y, h):```

#let compare = ```python
def compare_methods(n_eigenvalues=8, n_points=100):```

#let bisectionn = ```python
def bisection_eigenvalue(lambda_left, lambda_right, x_start, x_end, n_points, tol=1e-6, max_iter=50, shoot_func=shoot):```

#slide(title : [Another Good Example])[
  #code(title: [Methods])[
    #text(29pt)[
    #lastRunge
    #forwEuler
    #compare
    #bisectionn]
  ]
]

#slide(title: [Another Good Example])[
  #image("Images/Legendre.png")
  #image("Images/Legendre2.png")

  #text(28pt)[
    Lower the points get, better the difference between methods is:
    
  ]

  #grid(
  columns: (350pt, 290pt),
  rows: (auto),
  grid.cell(
    
    
    

    image("Images/difference.png", height: 6cm),
  
  ),
  grid.cell(
    image("Images/console.png")
  )

)
  
]

#focus-slide()[
  Thanks for your attention
  #text(fill: rgb("#041d4c"), size: 30pt)[

    Refs: \
  ]
  #text(24pt)[
    #link("https://pythonnumericalmethods.studentorg.berkeley.edu/notebooks/chapter23.02-The-Shooting-Method.html")[The-Shooting-Method - berkeley.edu]\
    #link("https://www.researchgate.net/publication/282523961_Numerical_Study_on_the_Boundary_Value_Problem_by_Using_a_Shooting_Method")[
      Numerical_Study_on_the_Boundary_Value
      Problem_by_Using_a_Shooting_Method
      ]
#link("https://www.math.iitb.ac.in/~siva/ma41707/ode7.pdf")[
      https://www.math.iitb.ac.in/~siva/ma41707/ode7.pdf
      ]
  ]
  


]