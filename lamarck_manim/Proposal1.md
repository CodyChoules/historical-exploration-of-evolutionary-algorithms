NIP2 Task 3: Machine Learning Project Proposal

Student Name: Cody Choules
Course: Introduction to Artificial Intelligence (C951)
Assessment: NIP2 Task 3
Date: 1/21/2026
Project Title: Pure Lamarckian Algorithm: Development, Visualization, and Comparative Analysis with Genetic and Lamarckian Genetic Algorithms


A.  Create a proposal for a machine learning project by doing the following:

1.  Describe an organizational need that your project proposes to solve.

This project addresses a critical commercial need for organizations facing complex optimization challenges where traditional Genetic Algorithms (GAs) and Lamarckian Genetic Algorithms (LGAs) may not provide optimal performance. Many businesses struggle with optimization problems in areas such as supply chain logistics, resource allocation, product design, manufacturing process optimization, and portfolio management where solutions require rapid adaptation and direct trait inheritance rather than slow genetic evolution. While LGAs have proven successful in specific domains like molecular docking (Morris et al., 1998), pure Lamarckian algorithms that operate entirely within the Lamarckian paradigm remain unexplored for commercial applications. This creates an opportunity gap where organizations may be missing superior optimization approaches for problems where acquired characteristics directly map to solution improvements. This project develops a commercial-grade pure Lamarckian algorithm with visualization capabilities that enables organizations to evaluate whether this paradigm offers competitive advantages for their specific optimization challenges, potentially reducing operational costs, improving resource utilization, and accelerating time-to-solution for complex business problems.

2.  Describe the context and background for your project.

This project operates within a commercial context where organizations require effective optimization solutions for complex business problems. The data consists of multi-dimensional optimization landscapes (mathematical functions like normal distributions, Rastrigin, Rosenbrock) that model real-world optimization challenges such as logistics routing, resource allocation, product configuration, and process optimization. These topologies represent fitness landscapes with local and global optima that mirror the complexity organizations face when seeking optimal solutions. Unlike traditional machine learning that learns from historical data, this project uses mathematical functions evaluated on-demand to simulate various business optimization scenarios. Lamarckian evolution, proposed by Jean-Baptiste Lamarck in the early 19th century, suggests organisms acquire traits during their lifetime that can be inherited. While largely rejected in biology, Lamarckian principles have demonstrated commercial success in computational optimization, particularly in molecular docking applications (Morris et al., 1998). The pure Lamarckian algorithm eliminates Darwinian mechanisms (no Malthusian mechanisms, no selection-based reproduction, no increase in reproduction based on fitness) to isolate Lamarckian characteristics for commercial evaluation. This project provides organizations with a new optimization tool that may outperform traditional approaches for problems where direct trait inheritance and rapid adaptation offer advantages over slow genetic evolution, potentially delivering measurable business value through improved solution quality, faster convergence, and better handling of specific problem characteristics.

3.  Review three outside works that explore machine learning solutions that apply to the need described in part A1.

Work 1: "An Examination of Lamarckian Genetic Algorithms" by Cameron Wellock and Brian J. Ross (GECCO 2001). This paper examines genetic algorithms where individuals undergo optimization before evaluation, implementing Lamarckian principles (Ross & Wellock, 2001). It shows how to compare evolutionary paradigms and provides implementation strategies for local optimization and trait encoding.

Work 2: "Automated Docking Using a Lamarckian Genetic Algorithm and an Empirical Binding Free Energy Function" by Garrett M. Morris et al. (Journal of Computational Chemistry, 1998). This paper demonstrates successful application of Lamarckian principles to molecular docking, showing real-world utility and validating the importance of understanding these mechanisms (Morris et al., 1998).

Work 3: "Evolutionary Algorithms in Theory and Practice" by Thomas Bäck (Oxford University Press, 1996). This textbook provides foundational theory including Eigen's quasispecies model and population dynamics equations that describe how evolutionary systems behave, enabling quantitative comparison of different paradigms (Bäck, 1996).

Describe how each reviewed work from part A3 relates to the development of your project.

Work 1 (Ross & Wellock, 2001) provides a comparative methodology framework that this project extends by comparing pure LA against both GA and LGA. The paper demonstrates how to compare evolutionary paradigms, informing the comparative analysis approach. Implementation strategies provide technical guidance adapted for pure Lamarckian mechanisms.

Work 2 (Morris et al., 1998) demonstrates practical commercial utility of Lamarckian principles, showing real-world business applications beyond theory. This supports commercial evaluation goals by providing concrete examples of Lamarckian success that demonstrate business value. The molecular docking success suggests potential commercial applications for pure LA across industries.

Work 3 (Bäck, 1996) provides theoretical foundations including population dynamics equations that help predict algorithm behavior. This guides algorithm design and enables quantitative comparison. The emphasis on isolating evolutionary mechanisms directly supports studying pure Lamarckian characteristics. Bäck's work was also important in understanding how to avoid GA mechanisms in the attempt to isolate the effects of LA, which proved challenging in practice.

Together, these works provide comparative methodology, practical commercial validation, and theoretical foundations needed for both technical implementation and commercial evaluation.



4.  Summarize the machine learning solution you plan to use to address the organizational need described in part A1.

The solution develops a pure Lamarckian algorithm that operates entirely within the Lamarckian paradigm. Organisms are represented as vectors where the origin is the initial state, the endpoint is the reproduction state after lifetime changes, and the vector difference represents inheritable acquired traits. Key mechanisms include: spawn origin generation (children spawn as mixtures of parent endpoints), habit vector (child actions tend toward parents' habits), purpose vector (gradient ascent toward goals), and child vector formation (combining purpose and habit with configurable weighting). Using Manim, the solution creates visualizations showing algorithm behavior, how LA approaches maxima and escapes local optima differently than GA/LGA, and comparative animations demonstrating distinct behavioral fingerprints. The algorithm is tested on multi-dimensional optimization landscapes and compared against standard GA and LGA implementations.

5.  Describe the benefits of your proposed machine learning

Commercial benefits: Provides organizations with a new optimization algorithm that may deliver superior performance for specific problem types, potentially reducing operational costs, improving resource utilization efficiency, and accelerating time-to-optimal-solution. Visualizations enable stakeholders to understand algorithm behavior and make informed decisions about algorithm selection for their specific business challenges.

Competitive advantages: Offers organizations access to an unexplored optimization paradigm that competitors may not be utilizing, potentially providing first-mover advantages in optimization-driven industries. The pure Lamarckian approach may excel in problems where acquired characteristics directly map to solution improvements, offering measurable ROI through better solutions.

Practical business value: Enables data-driven algorithm selection by demonstrating when pure Lamarckian approaches outperform traditional genetic algorithms or hybrid methods. Helps organizations identify optimization problems where LA provides competitive advantages, informing strategic technology decisions and potentially reducing dependency on slower or less effective optimization methods.

Broader commercial impact: Expands the optimization toolkit available to businesses, potentially improving outcomes across industries including logistics, manufacturing, finance, and product design. The implementation can be adapted for commercial use, and visualization capabilities support stakeholder communication and decision-making processes. Results can inform future commercial algorithm development and optimization strategy.



Machine Learning Project Design

B.  Describe your proposed machine learning project plan by doing the following:

1.  Define the scope of the proposed machine learning project.

In scope: Implementation of pure Lamarckian algorithm with core mechanisms (spawn origin, habit vector, purpose vector, child vector formation), visualization system using Manim for stakeholder communication, comparative implementation of baseline GA and LGA, testing on multiple multi-dimensional topologies representative of business optimization problems, analysis of algorithm behavior patterns and commercial performance metrics, documentation enabling commercial evaluation, and business case materials demonstrating value proposition.

Out of scope: Full production deployment infrastructure, integration with specific enterprise systems, extensive scalability testing beyond proof-of-concept, theoretical proofs of convergence, comprehensive user interface development beyond visualization rendering, enterprise security hardening, and production monitoring systems.

The project focuses on algorithm development, visualization, and comparative analysis for commercial evaluation purposes, prioritizing demonstration of business value and performance advantages over traditional approaches.

2.  Explain the goals, objectives, and deliverables for the proposed project.

Goals: Develop a commercial-grade pure Lamarckian algorithm isolating Lamarckian mechanisms from Darwinian components, create visualizations that effectively communicate algorithm performance and business value to stakeholders, provide comparative analysis that enables data-driven algorithm selection decisions, and demonstrate commercial viability of pure Lamarckian optimization approaches.

Objectives: Successfully implement pure LA with all core mechanisms suitable for commercial deployment, create Manim animations showing algorithm behavior and performance on 2D and 3D topologies representative of business problems, implement baseline GA and LGA for competitive comparison, quantitatively compare LA/GA/LGA across business-relevant metrics (convergence speed, solution quality, computational efficiency), and produce documentation enabling commercial evaluation and adoption decisions.

Deliverables: Production-ready pure Lamarckian algorithm implementation in Python, Manim visualization suite demonstrating algorithm performance and business value, comparative analysis report with business case evaluation, commercial deployment documentation, reproducibility package with all code and configuration files, and Jupyter notebook analysis supporting stakeholder decision-making.

3.  Explain how you will apply a standard methodology (e.g., CRISP-DM, SEMMA) to the implementation of your proposed project.

The project follows an adapted research methodology combining elements of CRISP-DM with software development practices:

Phase 1: Problem Understanding & Literature Review - Define need, review literature, establish theoretical foundations.

Phase 2: Algorithm Design - Design pure Lamarckian algorithm architecture, specify vector representation and mechanisms, design topology representation.

Phase 3: Implementation - Implement core LA algorithm, develop topology functions, implement gradient ascent for purpose vector, create Manim visualization framework, implement baseline GA and LGA.

Phase 4: Testing & Validation - Test on multiple topologies, validate mechanisms operate as designed, verify visualizations accurately represent behavior.

Phase 5: Comparative Evaluation - Run comparative experiments, collect metrics, analyze results, generate comparative visualizations.

Phase 6: Documentation & Communication - Document implementation and design decisions, create commercial evaluation materials, prepare visualizations, write comparative analysis report with business case, package deliverables.

This ensures systematic development while maintaining focus on both technical implementation and commercial value demonstration goals.

4.  Provide a projected timeline for the proposed project, including the start and end dates for each task.


Week 1 (Design & Core Implementation): December 16-22, 2025 - Complete literature review, finalize algorithm design, set up development environment (Python, Manim, JAX), implement topology generation and evaluation functions, implement spawn origin generation, habit vector calculation, and purpose vector (gradient ascent using JAX or sampling), basic testing.

Week 2 (Algorithm Integration & Visualization): December 23-29, 2025 - Integrate all mechanisms into complete LA algorithm, implement child vector formation, end-to-end testing, learn/refine Manim usage, create basic visualizations of LA behavior, develop comparative visualization framework.

Week 3 (Baseline Implementations & Comparative Experiments): December 30, 2025 - January 5, 2026 - Implement baseline Genetic Algorithm (GA) and Lamarckian Genetic Algorithm (LGA), ensure comparable experimental setup, run comparative experiments on multiple topologies, collect performance and behavior metrics, generate comparative visualizations.

Week 4 (Analysis & Refinement): January 6-12, 2026 - Complete statistical analysis, analyze results, refine visualizations based on findings, identify and document distinct behavioral fingerprints, prepare comparative analysis report.

Week 5 (Documentation & Finalization): January 13-19, 2026 - Complete commercial evaluation documentation, finalize all visualizations, package reproducibility materials, final review and polish, prepare submission package. The final product should be a commercial proof-of-concept demonstration built using ML and a number of available tools to help reproduce the results and enable business evaluation.

Total duration: 5 weeks (December 16, 2025 - January 19, 2026). Estimated effort: 50-70 hours total.



5.  List resources (e.g., hardware, software, work hours, third-party services) and all associated costs needed to implement the proposed solution.

Hardware: Personal computer/laptop (already available), 8GB RAM minimum, dedicated graphics card recommended for Manim rendering, ~10GB storage. Cost: $0.

Software: Python 3.9+ (free), Manim (free), JAX (free), NumPy (free), Matplotlib/Seaborn (free), Jupyter Notebooks (free), Git (free), Cursor IDE (free tier sufficient). Cost: $0.

Data: Multi-dimensional topologies generated programmatically, standard optimization test functions (publicly available). Cost: $0.

Work hours: Algorithm design and literature review (10-12 hours), core implementation (15-20 hours), visualization development (12-15 hours), comparative experiments and analysis (10-12 hours), documentation and finalization (8-10 hours). Total: 55-69 hours. Cost: $0.

Total project cost: $0. All resources are freely available open-source tools.

6.  Describe the criteria that you will use to evaluate the success of the project once it is completed.

Technical success: Pure LA successfully implements all core mechanisms and operates correctly on test topologies representative of business optimization problems, algorithm demonstrates distinct Lamarckian characteristics isolated from Darwinian mechanisms, Manim visualizations clearly demonstrate algorithm behavior and performance suitable for stakeholder presentation and commercial evaluation, implementation is documented and reproducible for commercial assessment.

Comparative analysis success: LA demonstrates measurably different behavioral fingerprint compared to GA and LGA, statistical comparison shows significant differences in at least two metrics (convergence speed, solution quality, escape patterns), analysis clearly identifies when and why LA behaves differently and when it provides commercial advantages.

Commercial success: Documentation and visualizations effectively communicate business value and performance advantages, materials enable stakeholders to make informed algorithm selection decisions, visualizations are understandable to both technical teams and business decision-makers. Stakeholders will understand how a Lamarckian optimization approach “learns” and how it differs from traditional GA approaches in commercial contexts.

Commercial evaluation success: All experimental results and visualizations are reproducible using provided code, methodology and findings are thoroughly documented for commercial evaluation, project provides actionable insights into pure Lamarckian optimization that enable informed business decisions about algorithm adoption.

Minimum acceptable: LA functions correctly, visualizations are created, basic comparison shows some differences from GA/LGA. Target: LA functions correctly with all mechanisms, high-quality visualizations suitable for stakeholder presentation, clear performance advantages identified and explained with business case implications, commercial evaluation materials complete.



Machine Learning Solution Design

C.  Describe the proposed machine learning solution you will use to address the organizational need identified in part A1 by doing the following:

1.  Identify the hypothesis of the proposed project.

A pure Lamarckian algorithm, operating entirely within the Lamarckian paradigm without Darwinian mechanisms, will function as an effective optimization algorithm and will demonstrate distinctly different behavioral characteristics and fingerprints compared to both standard Genetic Algorithms (GA) and Lamarckian Genetic Algorithms (LGA). The algorithm will successfully optimize multi-dimensional topologies through acquired trait inheritance rather than genetic operators, exhibit measurably different convergence patterns and local optima escape mechanisms, and the combination of spawn origin blending, habit vector inheritance, and need-driven gradient ascent will enable effective search of the topology space.

2.  Identify the machine learning algorithm(s) (i.e., supervised, unsupervised, or reinforcement learning) you will implement in your proposed solution.

Primary Algorithm: Pure Lamarckian Algorithm (LA) - Type: Evolutionary/Optimization Algorithm (Lamarckian paradigm), Learning Paradigm: Population-based optimization with acquired trait inheritance, Classification: Can be considered evolutionary computation or gradient-based optimization hybrid under the artificial life subdomain of AI, often utilized in reinforcement learning contexts. Thomas Bäck (1996) directly classifies evolutionary algorithms such as GA as "an example of an unsupervised learning technique, i.e. inductive learning by observation & discovery" (p. 33).

Comparative Algorithms: Genetic Algorithm (GA) - Standard Darwinian evolutionary algorithm for baseline comparison. Lamarckian Genetic Algorithm (LGA) - Hybrid approach combining GA with local search for comparison. Both could be considered under the classification of unsupervised and reinforcement learning.


a.  Justify the selection of the algorithm in part C2. Include one advantage and one limitation of the selected machine learning method.

Justification: Pure Lamarckian algorithms are rarely implemented in literature, creating opportunity for commercial competitive advantage. The approach allows evaluation of Lamarckian mechanisms in isolation from Darwinian components, tests whether pure Lamarckian principles can deliver superior performance for specific business optimization problems, and enables organizations to access an unexplored optimization paradigm. Recent commercial success of Lamarckian principles in molecular docking (Morris et al., 1998) demonstrates real-world value. Comparison with GA and LGA provides competitive baseline and shows how Lamarckian mechanisms perform when combined with Darwinian operators (Ross & Wellock, 2001), enabling informed commercial algorithm selection decisions.

Advantage: Isolates Lamarckian mechanisms for commercial evaluation. May excel in business problems where acquired traits directly map to solution improvements, potentially delivering superior performance and ROI. Provides clear model of "form follows function" principle applicable to optimization-driven industries. Offers competitive advantage through access to unexplored optimization paradigm.

Limitation: May converge slower than GA/LGA due to lack of genetic diversity mechanisms. Limited exploration capability compared to population-based genetic algorithms. May struggle with highly multi-modal landscapes without crossover-based exploration. Less studied than GA/LGA, so fewer established best practices. The algorithm is heuristic and does not guarantee globally optimal solutions.

3.  Describe the tools and environments that will be used to develop the proposed machine learning solution, including any third-party code.

Development environment: Windows 10/11, Linux, or macOS with Python 3.9+, Cursor IDE or preferred editor.

Core libraries: Manim (Mathematical Animation Engine) for creating algorithm behavior animations and production-quality mathematical visualizations. JAX for automatic differentiation and gradient computation needed for purpose vector calculation, providing efficient numerical computing with GPU acceleration support. NumPy for numerical computation, topology evaluation, and vector operations. Matplotlib/Seaborn for additional visualization support. Jupyter Notebooks for interactive development and documentation.

Supporting tools: Git for version control, SciPy for statistical analysis if needed, Pandas for experimental results analysis.

Third-party code: Manim Community Edition (MIT license) for all visualizations. JAX (Apache 2.0 license) for gradient computation. Standard optimization test functions (Rastrigin, Rosenbrock, etc.) publicly available for topology generation.

Development workflow: Algorithm development in Python modules, interactive experimentation in Jupyter notebooks, visualization scripting using Manim, batch experiments for comparative analysis, integrated documentation in Jupyter notebooks.

Deployment: Local execution for development and testing, Manim rendering produces video files suitable for presentation. All tools are open-source and freely available.

4.  Explain the process you will use to measure the performance of your proposed machine learning solution.

Performance measurement evaluates the pure Lamarckian algorithm and compares it against GA and LGA across multiple dimensions. Fitness is measured as topology value at the organism's final position.

Primary metrics: Convergence metrics (generations to reach fitness threshold, fitness improvement rate, final fitness achieved), local optima handling (success rate escaping local optima, number of local optima visited, diversity of explored regions), population/system dynamics (population diversity over time, convergence behavior patterns, exploration vs. exploitation balance), behavioral fingerprint metrics (trajectory patterns, distribution of final solutions, sensitivity to initial conditions, response to different topology characteristics).

Evaluation methodology: Test on various function types (normal distributions, multi-modal functions, functions with ridges and valleys), run each algorithm 20-30 times per topology with different random seeds, conduct side-by-side comparison with statistical tests to identify significant differences, calculate effect sizes, create animated trajectories showing how each algorithm explores the topology, generate convergence curve comparisons and population distribution visualizations, identify distinct behavioral patterns and characterize fingerprint differences.

Success criteria: LA successfully optimizes test topologies, clear behavioral differences identified between LA/GA/LGA, differences are statistically significant and explainable, visualizations effectively demonstrate distinct characteristics. Visualization metrics include quality and clarity of Manim animations, effectiveness in communicating paradigm differences, suitability for stakeholder presentation and commercial evaluation, accuracy in representing algorithm behavior.

Description of Data Set

D.  Describe the data for your proposed project by doing the following:

1.  Identify the source(s) of the data for your proposed project.

The project uses synthetic multi-dimensional topologies (mathematical functions) rather than traditional datasets. These topologies represent optimization landscapes typical of problems addressed by evolutionary algorithms.

Primary data source: Programmatically generated mathematical functions that define optimization landscapes for algorithm testing. Format: Python functions that evaluate fitness at given coordinates.

Topology types: Normal/Gaussian distributions (smooth, single-peak functions with maximum at origin, configurable variance and dimensionality), standard optimization test functions (Rastrigin function - highly multi-modal with many local optima, Rosenbrock function - valley-shaped with narrow global optimum, Ackley function - many local optima with moderate difficulty, Sphere function - simple convex baseline), custom topologies (combinations of multiple functions, functions with specific characteristics like steep gradients, plateaus, ridges).

Data characteristics: Dimensionality of 2D for visualization, higher dimensions for testing. Configurable coordinate ranges (e.g., [-5, 5] per dimension). Continuous functions evaluated at any point in space. Deterministic - same coordinates always yield same fitness value.

All topologies are synthetic, allowing controlled experimentation and clear visualization while representing problem types typical of evolutionary algorithm applications.

2.  Describe the data collection method.

The data collection method is programmatic generation of mathematical topologies using Python functions. Collection process: Define mathematical functions in Python that represent fitness landscapes, set parameters (dimensions, ranges, difficulty) for each topology type, evaluate functions on-demand during algorithm execution at requested coordinates. Topologies are not stored as datasets but computed as needed.

Data characteristics: Type is mathematical functions (not tabular data), collection method is programmatic generation (synthetic), purpose is algorithm testing and visualization, format is Python callable functions.

a.  Discuss one advantage and one limitation of the data collection method described in part D2.

Advantage: Complete control over topology characteristics enables systematic testing of algorithm behavior under different conditions (smooth vs. rugged, single vs. multi-modal, etc.). This allows isolation of factors affecting algorithm performance. Mathematical functions are perfectly reproducible - same coordinates always yield same values with no data collection variability or missing values. Synthetic 2D/3D topologies can be easily visualized, making algorithm behavior clear and suitable for stakeholder communication. No privacy concerns since no real-world data is used. Can generate as many evaluation points as needed without storage constraints.

Limitation: Synthetic topologies may not capture all complexities of real-world optimization problems (e.g., noisy evaluations, constraint interactions, problem-specific structures). Results may not directly generalize to practical applications. Mathematical functions are abstractions that may oversimplify real problem characteristics, potentially missing important factors that affect algorithm performance in practice.

Despite these limitations, synthetic topologies are appropriate for understanding algorithm mechanisms and comparative analysis, which are the primary goals of this commercial evaluation project.

3.  Explain how you will prepare your data for use by the machine learning algorithm(s) from part C2 for your proposed project, including data set formatting, missing data, outliers, dirty data, or mitigation of other data anomalies.

Data preparation involves: 
Step 1 - Function definition and implementation: Implement topology functions in Python (normal distributions, standard test functions, custom functions), ensure functions are vectorized for efficient evaluation, validate function correctness by testing known optima and boundary behavior. 
Step 2 - Parameter configuration: Define test suite with varying characteristics (different dimensionalities, difficulty levels, modality), create configuration files specifying topology parameters. 
Step 3 - Evaluation framework: Design efficient evaluation system with batch evaluation capabilities, set up gradient computation for JAX-based purpose vector. 
Step 4 - Visualization preparation: Prepare topology visualization (2D contour plots, 3D representations for Manim, color mapping for fitness values). Step 5 - Algorithm integration: Ensure topology functions integrate with algorithms (compatible input/output formats, efficient evaluation, support for gradient computation).

Handling data issues: Validate all functions return valid numeric values (no NaN, Inf), handle edge cases (evaluation at boundaries, extreme values), ensure numerical precision is adequate, optimize function evaluation for speed (vectorization, JAX compilation).

Data quality assurance: Test functions on known points to verify correctness, compare implementations against reference implementations of standard functions, validate that optima locations match expected values, check that function characteristics match intended design.

Special considerations: For need vectors (representing the organism seeking a need), ensure functions are differentiable and JAX-compatible, or implement sampling-based gradient approximation. Ensure topology representations work with Manim visualization requirements. Document all function parameters and ensure random seeds are controlled for reproducibility.

4.  Describe behaviors that should be exercised when working with and communicating about sensitive data in your project.

Since this project uses synthetic, programmatically generated mathematical functions rather than real-world data, sensitive data concerns are minimal. However, appropriate practices should be exercised:

Data privacy and security: All topologies are synthetic mathematical functions with no connection to real-world sensitive information. Clearly document that all data is programmatically generated for research purposes. Ensure topology generation is reproducible and documented, enabling verification that no hidden real data is included.

Communication practices: Always specify that topologies are synthetic mathematical functions, not real-world data. Do not imply that results apply directly to specific real-world problems without validation. Document topology generation methods and cite any standard test functions used.

Ethical considerations: Properly cite any standard test functions or benchmark problems used. Enable others to reproduce topologies and results. Accurately represent limitations of synthetic data in generalizing to real-world applications.

Since this project uses only synthetic mathematical functions, the primary responsibility is clear communication about data nature and proper citation of any standard test functions or methodologies used.



E.  Acknowledge sources, using in-text citations and references, for content that is quoted, paraphrased, or summarized.

E.1 AI Assistance Disclosure

In accordance with WGU's "Use of Artificial Intelligence (AI) Tools" policy (Western Governors University, 2025), the following AI tools were used in the preparation of this submission:

Cursor AI (Composer LLM Model) was used for permitted purposes including:
- Explanation of concepts and helping to structure the proposal
- Correcting spelling, grammar, and punctuation
- Enhancing clarity, sentence structure, and fluency of language
- Reviewing tone or style suggestions for improving readability
- Summarizing verbose explanations
- Assisting with APA citation formatting

Grammarly was used for permitted purposes including:
- Correcting spelling, grammar, and punctuation
- Enhancing clarity, sentence structure, and fluency of language
- Reviewing tone or style suggestions for improving readability

All use of AI assistance in this document is meant to comply with WGU guidelines and policies on AI usage as of the submission date. This use demonstrates critical thinking and original thought, and does not replace performance and development of core academic skills as outlined in the WGU policy.


E.2 References

Bäck, T. (1996). Evolutionary algorithms in theory and practice: Evolution strategies, evolutionary programming, genetic algorithms (1st ed.). Oxford University Press.

Cursor AI. (2025). Composer [Large language model]. https://cursor.sh/

Grammarly Inc. (2025). Grammarly [Software]. https://www.grammarly.com/

Morris, G. M., Goodsell, D. S., Halliday, R. S., Huey, R., Hart, W. E., Belew, R. K., & Olson, A. J. (1998). Automated docking using a Lamarckian genetic algorithm and an empirical binding free energy function. Journal of Computational Chemistry, 19(14), 1639-1662. https://doi.org/10.1002/(SICI)1096-987X(19981115)19:14%3C1639::AID-JCC10%3E3.0.CO;2-B

Ross, B. J., & Wellock, C. (2001). An examination of Lamarckian genetic algorithms. Proceedings of the Genetic and Evolutionary Computation Conference (GECCO 2001). https://www.cosc.brocku.ca/~bross/research/gecco_2001_lbp.pdf

Western Governors University. (2025, November 7). Use of Artificial Intelligence (AI) Tools. WGU Student Policy Handbook. https://cm.wgu.edu/t5/WGU-Student-Policy-Handbook/Use-of-Artificial-Intelligence-AI-Tools/ta-p/67811




