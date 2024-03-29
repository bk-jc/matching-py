IGNORED = [
    " Gas",
    "ATM",
    "Aerobic",
    "BMD",
    "Cadence",
    "Clarify",
    "Cocoa",
    "Cooking",
    "Tennis",
    "Spice",
    "Swimming",
    "Swing",
    "Diving",
    "Focus",
    "Golf",
    "Standards",
    "Installer",
    "Foreman",
    "Storage",
]
DUPLICATES = [
    ("AngularJS", "Angular"),
    ("ASP", "ASP.net"),
    ("Audio Video Production", "Audio and video postproduction"),
    ("XML", "XML Parsing (SAX/DOM)"),
    ("Windows", "Windows 10", "Windows 7", "Windows 95/98/Me", "Windows CE", "Windows NT", "Windows XP/2000/Vista"),
    ("Textile Industry", "Textilindustrie"),
    ("Surgery", "Surgery Technique", "General Surgery"),
    ("Bricklayer / Plasterer", "Stucco Plasterer"),
    ("Nursing", "Nursing examination"),
    ("IT - Information Technology", "IT General Skills"),
]

MAPPING = {
    " Gas": "Gas",
    ".net": ".NET",
    "3D Design": "",
    "3D Studio Max": "Autodesk 3ds Max",
    "ABACUS": "Abacus (Financial software)",
    "ABAP": "ABAP (Advanced Business Application Programming)",
    "ACCA": "ACCA (Association of Chartered Certified Accountants)",
    "ADT": "ADT (Android Development Tools)",
    "AFP": "AFP (Advanced Function Presentation)",
    "AIX": "AIX (Advanced Interactive eXecutive)",
    "APOP": "APQP (Advanced Product Quality Planning)",
    "ARIS": "ARIS (Architecture of Integrated Information Systems)",
    "ASI-Bus": "AS-Interface Bus",
    "ASP": "",
    "ASP.net": "",
    "ATM": "",
    "AUTOSAR": "AUTOSAR (AUTomotive Open System ARchitecture)",
    "Accident Surgery": "",
    "Account Receivable / Payable": "Accounts Receivable and Payable Management",
    "Accounting": "",
    "Actuarial Mathematikc": "Actuarial Mathematics",
    "Acupuncture": "",
    "Ada": "Ada (programming language)",
    "Administrative Law": "",
    "Adobe Creative Suite": "",
    "Adobe Experience Manager": "",
    "Adobe Illustrator": "",
    "Adobe InDesign": "",
    "Adobe PageMaker": "",
    "Adobe Photoshop": "",
    "Adobe Premiere": "",
    "Adonis": "ADONIS (BPM tool)",
    "Advertisung Technology": "Advertising Technology",
    "Aerobic": "",
    "Aerodynamic": "",
    "Aeronautics": "",
    "After Effects": "",
    "Agil PLM": "Agile Product Lifecycle Management",
    "Agile Software Devopment": "Agile Software Development",
    "Agriculture Technical Assistance": "",
    "Agriculture and Forestry": "",
    "Agronomy": "",
    "Aircraft Mechanic": "",
    "Ajax": "Ajax (programming)",
    "Alternative Medicine": "",
    "Amadeus": "Amadeus Global Distribution System",
    "Analogue technique": "",
    "Anaplasty": "",
    "Anatomy": "",
    "Android": "",
    "Andrology": "",
    "Anesthesia": "Anesthesiology",
    "Angiology": "",
    "Angular": "",
    "AngularJS": "",
    "Ansys": "ANSYS (Software)",
    "Antenna Technology": "",
    "Anthropologie": "Anthropology",
    "Anti Money Laundering": "Anti-Money Laundering",
    "Apache Lucene": "",
    "Apache Maven": "",
    "Apache Tomcat": "",
    "Apache Webserver": "",
    "Apache-Axis": "Apache Axis",
    "Apache-FOP": "Apache FOP",
    "Apiarist": "Apiary",
    "AppDynamics": "",
    "Apple iOS": "",
    "Archeology": "",
    "ArchiCAD": "",
    "Architecture": "",
    "Art History": "",
    "Artificial Intelligence": "",
    "Assembler": "Assembler Language Programming",
    "Asset Accounting": "",
    "Asset and Liability Management": "",
    "Astronomy": "",
    "Audio Video Production": "Audio and Video Production",
    "Audio and video postproduction": "",
    "Audiology": "",
    "Auditing": "",
    "AutoCAD": "AutoCAD (Software)",
    "AutoDesk": "",
    "Automation Engineering": "",
    "Automotiv Merchant": "Automotive Merchandising",
    "Automotive Engineering": "",
    "Automotive Industry": "Automotive Industry Expertise",
    "Aviation / Defense": "",
    "Avionik": "Avionics",
    "BEA Tuxedo": "",
    "BEA WebLogic Server": "",
    "BGP": "BGP (Border Gateway Protocol)",
    "BI / DWH (Data Warehouse)": "Business Intelligence and Data Warehousing",
    "BMC Patrol": "BMC Patrol (Software)",
    "BMD": "",
    "BPM - Business Process Management": "BPM (Business Process Management)",
    "BSI - IT-Basic Protection": "BSI (IT-Basic Protection)",
    "Backup and archiving systems": "",
    "Baker": "Baking",
    "Balance sheet accounting": "",
    "Balance sheet analyses": "Balance sheet analysis",
    "Bank Accounting": "",
    "Bank Law": "Banking law",
    "Banking": "",
    "Banking Core Systems": "",
    "Banking know-how": "Banking Expertise",
    "Barista": "",
    "Bath Attendant": "",
    "Beer / Beverage Technology": "",
    "BigData": "Big Data",
    "Biochemistry": "",
    "Bioinformatics": "",
    "Biological Technical Assistance": "",
    "Biology": "",
    "Biomedicine": "",
    "Biometric": "Biometrics",
    "Biopharmacy": "",
    "Biostatistic": "Biostatistics",
    "Biotechnology": "",
    "Boat Building": "",
    "Bookbinder": "Bookbinding",
    "Bottomer / Floorer / Tiler": "Flooring / Tiling",
    "Brazer": "Brazing",
    "Bricklayer / Plasterer": "Plastering",
    "Bridge Building": "",
    "Broadcasting": "",
    "Building Facilities": "",
    "Building Law": "",
    "Building Mechanics": "",
    "Building Physics": "",
    "Building Services Engineering": "",
    "Building trade": "",
    "Business Economics": "",
    "Business Objects": "",
    "Butcher": "",
    "C": "C (programming language)",
    "C#": "C# (programming language)",
    "C++": "",
    "CAD/CAM": "",
    "CADdy": "CADdy (software)",
    "CAIA": "CAIA (Chartered Alternative Investment Analysis)",
    "CAN-Bus": "CAN-Bus (Controller Area Network)",
    "CAS - Computer Aided Styling": "Computer Aided Styling (CAS)",
    "CATIA CADAM": "",
    "CFA": "CFA (Chartered Financial Analysis)",
    "CIA": "CIA (Confidentiality, Integrity, and Availability)",
    "CIMA": "CIMA (Chartered Institute of Management Accountants)",
    "CLC Development": "",
    "CMMI": "CMMI (Capability Maturity Model Integration)",
    "COBIT": "COBIT (Control Objectives for Information and Related Technologies)",
    "COM/OLE/ActiveX": "",
    "CORBA/IIOP": "",
    "CPA": "CPA (Certified Public Accountant)",
    "CRM - Customer Relationship Mgmt": "Customer Relationship Management (CRM)",
    "CSS": "CSS (Cascading Style Sheets)",
    "CVS": "",
    "Cabinet Maker": "Cabinet making",
    "Cadence": "",
    "Canalization": "",
    "Car Mechanik": "Car Mechanics",
    "Cardiology": "",
    "Cardiosurgery": "",
    "Carpenter": "Carpentry",
    "Carrossier": "Carrosserie",
    "Cartel Law/Antitrust Law": "",
    "Caseation": "",
    "Cash Management": "",
    "Caster": "",
    "Catering": "",
    "Catia V4/V5": "",
    "Ceramist": "",
    "Chamfer": "",
    "Change Management": "",
    "Chemical Engineering": "",
    "Chemical Technical Assistance": "",
    "Chemistry": "",
    "Chiropractic": "Chiropractics",
    "Chromatography": "",
    "Cinema 4D": "",
    "Cisco": "Cisco (Software0",
    "Citrix": "Citrix (Software)",
    "Civil Engineering": "",
    "Civil Law": "",
    "Clarify": "",
    "Cleantech": "Clean technology",
    "Climatology": "",
    "Clipper": "",
    "Clockmaker": "",
    "Cloud Computing": "",
    "Cobbler": "Cobblery",
    "Cobol": "COBOL (Programming Language)",
    "Cocoa": "",
    "Cognitive Science": "",
    "Cognos": "Cognos (Software)",
    "Cold Fusion (CFML)": "",
    "Collective Bargaining Law": "",
    "Commercial Law": "",
    "Communications Engineering": "",
    "Company Law/ Corporate Law": "Company Law and Corporate Law",
    "Computer Science": "",
    "Computer Tomography": "Computed Tomography",
    "Concrete Building": "",
    "Confectioner": "Confectionery",
    "Configuration Management": "",
    "Construction Economy": "",
    "Construction Industry": "Construction Industry Expertise",
    "Construction Machine License": "",
    "Construction Mechanics": "",
    "Content management": "",
    "Control / Measurement Technology": "",
    "Cooking": "",
    "Corel Draw": "CorelDRAW",
    "Corporate Finance": "",
    "Cosmetician": "Cosmetics",
    "Cost Accounting": "",
    "Crane License": "",
    "Criminal Law": "",
    "Criminology": "",
    "Crystal Reports": "SAP Crystal Reports",
    "Cutting technique/ CNC": "CNC Machining",
    "Cybernetics": "",
    "DB2": "IBM Db2 (Software)",
    "DCOM": "Distributed Component Object Model (DCOM)",
    "DOORS": "Rational DOORS (IBM)",
    "DSpace": "",
    "DameWare": "",
    "Data Management": "",
    "Data Mining": "",
    "Data Modeling": "",
    "Datev": "",
    "Defense and Recovery": "",
    "Delphi": "",
    "Dental Assistant": "",
    "Dental Hygienists": "",
    "Dental Technology": "",
    "Dentistry": "",
    "Derby": "",
    "Dermatology": "",
    "DevOps": "",
    "Diabetology": "",
    "Dialysis": "",
    "Diet chef": "",
    "Dietetics": "",
    "Dietician": "",
    "Digger License": "",
    "Digital Technique": "",
    "Direct Marketing": "",
    "Distributed Systems": "",
    "Diving": "",
    "Django": "",
    "Docker": "",
    "Document management": "",
    "Documentum": "",
    "Draftsight": "",
    "Dreamweaver": "",
    "Drive Engineering": "",
    "Drive Simulation": "",
    "Driving Instruction License": "",
    "Driving Licence B Cars": "Driving",
    "Driving License A Motorcycles": "Motorcycle driving",
    "Driving License BE Cars with Trailer": "Driving with trailer",
    "Driving License C Trucks": "Truck driving",
    "Driving License C1E Light Trucks with Trailer": "Light truck driving with trailer",
    "Driving License CE Trucks with Trailer": "Truck driving with trailer",
    "Driving License D Buses": "Bus driving",
    "Driving License DE Bus with Trailer": "Bus driving with trailer",
    "Driving License Railway Engine": "Railway driving",
    "Driving License T Tractors": "Tractor driving",
    "Drupal": "",
    "Dry Walling": "",
    "Dynatrace": "",
    "E-Balance": "",
    "E-Learning": "",
    "EAI - Enterprise Application Integration": "",
    "EDGE": "",
    "EDI - Electronic Data Interchange": "",
    "EJB": "",
    "EMV": "",
    "EPLAN": "",
    "ER Model": "",
    "ERP Systems": "Enterprise resource planning (ERP)",
    "ERwin": "",
    "ESQL": "",
    "ETL - Extract/Transform/Load": "",
    "Eagle": "",
    "Easylohn": "Easylohn (Software)",
    "Eclipse": "",
    "Economic Geography": "",
    "Economic Pedagogy": "",
    "Economic Psychology": "",
    "Economic Science": "Economics",
    "Economic Sociology": "",
    "Education": "",
    "Education Sciences": "",
    "Electrician": "Electrical wiring",
    "Electromechanics": "",
    "Electronic / Electrical Engineering": "Electrical engineering",
    "Embedded Software Development": "",
    "Embroiderer": "Embroidery",
    "Emission Technology/Exhaust Technology": "",
    "Employer Branding": "",
    "Employment Law": "",
    "Endocrinology": "",
    "Endoskopy": "Endoscopy",
    "Energy Technology": "",
    "Energy and Utility Industry": "",
    "Engine Engineering": "",
    "Engraver": "Engraving",
    "Enterprise Content Management": "",
    "Enterprise Service Bus": "",
    "Essbase": "",
    "Ethernet": "",
    "Excel": "Microsoft Excel",
    "Export Know-how": "Export Expertise",
    "Extreme Programming": "",
    "FEM - Finite Element Method": "FEM (Finite Element Method)",
    "FMEA / FMECA": "",
    "Familiy Law": "",
    "FileMaker Pro": "",
    "FileNet": "",
    "Filtration": "",
    "Final Cut": "",
    "Finance": "",
    "Finance Science": "",
    "Financial Manager": "",
    "Financial Mathematics": "",
    "Financial Planning and Analysis (FP&A)": "",
    "Financing": "",
    "Fine Mechanics": "",
    "Fire Protection Engineering": "",
    "Firewall": "",
    "Flash/ActionScript": "",
    "Flexray": "",
    "Florist": "",
    "Fluid Dynamics": "",
    "Focus": "",
    "Fonds Accounting": "",
    "Food Hygiene": "",
    "Food Hygiene Regulation": "",
    "Food Production": "",
    "Food Technology": "",
    "Foreman": "",
    "Forensic": "Forensics",
    "Forensic Medicine": "",
    "Forklift License": "Forklift Driving",
    "Fortran": "",
    "Frame Relay": "",
    "Frontpage": "",
    "Furrier": "",
    "GIMP - GNU Image Manipulation Program": "",
    "GPRS": "",
    "Galileo": "",
    "Galvanik": "",
    "Gardener": "Gardening",
    "Gastroenterology": "",
    "Gear Technique": "",
    "General Knowledge": "",
    "General Medicine": "",
    "General Surgery": "",
    "Genetic engineering": "",
    "Genetics": "",
    "Geodesy": "",
    "Geography": "",
    "Geoinformatics": "",
    "Geology": "",
    "Geomatics": "",
    "Geophysicist": "Geophysics",
    "Geotechnics": "Geotechnical engineering",
    "Geriatric Medicine": "",
    "Git": "",
    "Glas Blower": "Glass blowing",
    "Glass Construction": "",
    "Glazier": "",
    "Gold/Silversmith": "",
    "Golf": "",
    "Good Manufacturing Practice": "Manufacturing Expertise",
    "Google Web Toolkit": "",
    "Grails": "",
    "Groovy": "",
    "Groupware/Office Communication": "",
    "Gymnastics": "",
    "Gynecology": "",
    "HRM - Human Resource Management": "",
    "HTML/XHTML": "",
    "HVAC Technique": "Heating, ventilation, and air conditioning (HVAC)",
    "Hair Dresser": "Hairdressing",
    "Handcraft": "",
    "Hardware Technique": "",
    "Hatter": "",
    "Health Care": "",
    "Hearing aid acoustic": "",
    "Heat Engineering / Energy Engineering": "",
    "Heavy Current Technique": "",
    "Helios": "",
    "Hematology": "",
    "Hibernate": "",
    "Hicad": "",
    "Histology": "",
    "Hotel Management": "",
    "Hudson": "",
    "Human Genetics": "",
    "Human Resources": "",
    "Humanities knowledge": "",
    "Hydraulic": "",
    "Hydraulic Engineering": "",
    "Hydrology": "",
    "Hygiene Medicine": "",
    "Hyper-V": "",
    "Hyperion": "",
    "I2C-Bus": "",
    "IAS": "",
    "IBM CICS": "",
    "IBM Lotus (Notes/Domino/...)": "IBM Lotus (Software",
    "IBM MQ-Series": "",
    "IBM Mainframe": "",
    "IBM Tivoli": "",
    "IBM WebSphere": "",
    "IBM-SNA": "",
    "IDMS": "",
    "IEC standards": "",
    "IFRS - International Financial Reporting Standards": "International Financial Reporting Standards (IFRS)",
    "IFRS 9": "",
    "IMS/DB": "",
    "IPMA -  Intern. PM Association": "International Project Management Association (IPMA)",
    "IPv4/IPv6": "",
    "ISAM": "",
    "ISO 13209": "",
    "ISO 14001": "",
    "ISO 20000": "",
    "ISO 26262": "",
    "ISO 27001": "",
    "ISO 31000": "",
    "ISO 9001": "",
    "ISTQB - Certified Tester": "",
    "IT - Information Technology": "Information technology (IT)",
    "IT Distribution": "",
    "IT General Skills": "",
    "IT Purchase": "",
    "IT-Consulting/IT-Support": "IT Support",
    "IT-EDPC": "",
    "IT/ Telecommuncation": "IT / Telecommunication",
    "ITIL": "Information Technology Infrastructure Library (ITIL)",
    "Immunology": "",
    "Incident Management": "",
    "Income Tax Law": "",
    "Industrial Ceramist": "",
    "Industrial Clerk": "",
    "Industrial Construction": "",
    "Industrial Management": "",
    "Industrial Mechanics": "",
    "Industrial engineering": "",
    "Industrie Know-how": "Industry Expertise",
    "Infectiology": "",
    "Informatica": "",
    "Information Engineering": "",
    "Information Technology": "",
    "Informix": "",
    "Inheritance Law": "",
    "Injection Technology": "",
    "Insonvency Law": "",
    "Installer": "",
    "Insurance industry": "",
    "Insurance salesman": "",
    "IntelliJ IDEA": "",
    "Intensive Care": "",
    "Interactive Media": "",
    "Intergraph": "",
    "Interior Design": "",
    "Internal Medicine": "",
    "Internet": "",
    "Inventor": "",
    "Investment Banking": "",
    "Ironer": "",
    "JAAS": "",
    "JBoss": "",
    "JCL": "",
    "JDBC": "",
    "JMS": "",
    "JNI": "",
    "JQuery": "",
    "JSF": "",
    "JSP": "",
    "JUnit": "",
    "Jakarta-Ant": "",
    "Jakarta-JMeter": "",
    "Jakarta-Struts": "",
    "Jakarta-Tapestry": "",
    "Jakarta-Velocity": "",
    "Jasper Report": "",
    "Java": "Java (Programming Language)",
    "Java SE": "",
    "Java/J2EE": "",
    "JavaMail": "",
    "JavaScript": "",
    "Jenkins": "",
    "Jeweler": "",
    "Joiner": "Joinery",
    "Joomla": "",
    "Journalism": "",
    "Judiciaries": "",
    "Jurisprudence": "",
    "Kaizen": "",
    "Kartography": "",
    "Kinematics": "",
    "Kinesiology": "",
    "Kinetic": "",
    "Knowledge Management": "",
    "LDAP": "",
    "LTE": "",
    "LabVIEW": "",
    "LabWindows": "",
    "Lacquerer": "Lacquering",
    "Laminating": "",
    "Landscaping": "",
    "Laravel": "",
    "Lathe Engineer": "Lathe Engineering",
    "Lawyer": "",
    "Leasing": "",
    "Legal Assistance": "",
    "Life Guard": "",
    "Light Engineering": "",
    "Linguistics": "",
    "Linq": "",
    "Linux": "Linux (Operating System)",
    "Lisp": "",
    "Lithography": "",
    "Llaboratory Medicine": "",
    "Locksmith": "Locksmithing",
    "Log4J": "",
    "Logistics": "",
    "Logpaedics": "",
    "Lotus Notes": "IBM Lotus Notes (Software)",
    "Lotus Notes DB": "IBM Lotus Notes DB (Software)",
    "Low PowerTechnology": "",
    "Low Voltage Engineering": "",
    "MDA - Model Driven Architecture": "",
    "MOST-Bus": "",
    "MPLS": "",
    "MQ Telemetry Transport": "",
    "MS Active Directory": "Microsoft Active Directory",
    "MS Dynamics AX (Axapta)": "Microsoft Dynamics AX (Axapta)",
    "MS Dynamics CRM": "Microsoft Dynamics CRM",
    "MS Dynamics GP": "Microsoft Dynamics GP",
    "MS Dynamics NAV (Navision)": "Microsoft Dynamics 365 Business Central",
    "MS Exchange Server": "Microsoft Exchange Server",
    "MS Office": "Microsoft Office",
    "MS SQL Server": "Microsoft SQL Server",
    "MS Sharepoint Portal Server": "Microsoft Sharepoint",
    "MS Windows Server": "Microsoft Windows Server",
    "MS-Access": "Microsoft Access",
    "MS-Project": "Microsoft Project",
    "MS-Visio": "Microsoft Visio",
    "MS-Word": "Microsoft Word",
    "MSCA": "",
    "MVS/OS-390": "",
    "Mac OS": "macOS",
    "Machine Learning": "",
    "Magnetic Resonance Imaging": "",
    "Manicure/Pedicure": "",
    "MariaDB": "",
    "Marketing Software": "",
    "Marketing/Market Research/Advertising": "Marketing",
    "Materials Science": "",
    "Mathematics": "",
    "Matlab/Simulink": "MATLAB/Simulink (Software)",
    "Maya": "",
    "Mechanics": "",
    "Mechatronics": "",
    "Media Computer Science": "",
    "Media Engineering": "",
    "Media Studies": "",
    "Media/Information Professional": "",
    "Media/Multimedia/Press": "",
    "Medical Computer Science": "",
    "Medical Law": "",
    "Medical Professional": "",
    "Medical Science/ Physicians": "",
    "Medical Technical Assistance": "",
    "Medical Technology": "",
    "Mercury LoadRunner": "",
    "Mercury WinRunner/QualityCenter": "",
    "Mesotherapy": "",
    "Metal Machining": "",
    "Metallurgy": "",
    "Meteorology": "",
    "MicroStation": "",
    "Microelectronics": "",
    "Microsoft Azure": "",
    "Microsoft Intune": "",
    "Microsoft Lync": "",
    "Middleware/Transactions": "",
    "Midia Law": "",
    "Miller": "",
    "MindManager": "",
    "Mineral Oil": "",
    "Mineralogy": "",
    "Mining": "",
    "Model View Controller": "",
    "Modula": "",
    "Molecular Biology": "",
    "Motor Mechanic": "Motor Mechanics",
    "Motor Technique": "",
    "Motorbike Mechanic": "",
    "Multimedia": "",
    "Municipal Affairs": "",
    "Museology": "",
    "MySAP-ERP": "",
    "MySql": "",
    "NUnit": "",
    "NVH": "",
    "Nano Technics": "",
    "Nemetschek": "",
    "Neo4j": "",
    "Neonatology": "",
    "Nephrology": "",
    "NetBeans": "",
    "Network": "",
    "Network Engineering": "",
    "Neurlogy": "",
    "Neurosurgery": "",
    "NoSQL": "",
    "Notary": "",
    "Notary Clerk": "",
    "Novell": "",
    "Novell GroupWise": "",
    "Nuclear Technology": "",
    "Nursing": "",
    "Nursing Management": "",
    "Nursing Pedagogy": "",
    "Nursing Science": "",
    "Nursing examination": "",
    "Nutrition Science": "",
    "ODBC": "",
    "OOA/OOD": "",
    "OPC Unified Architecture": "",
    "OS/400": "IBM i (Operating System)",
    "ObjectiveC": "Objective-C",
    "Occupational Medicine": "",
    "Occupational Therapy": "",
    "Office Tools": "",
    "Offiice Communication Professional": "Office Communication Professional",
    "Oncology": "",
    "Online Marketing": "",
    "Open Cast Mining": "",
    "OpenFoam": "",
    "OpenOffice.org Base": "",
    "Ophthalmic Optics": "",
    "Ophthalmology": "",
    "Optician": "",
    "Oracle": "",
    "Oracle Application Server": "",
    "Oracle E-Business Suite": "",
    "Oracle Forms": "",
    "Oral and Maxillofacial Surgery": "",
    "Organ Building": "",
    "Orthomolecular Medicine": "",
    "Orthopedagogy": "",
    "Orthopedy": "",
    "Osteopathy": "",
    "Otolaryngology": "",
    "Outlook": "Microsoft Outlook",
    "PDF": "PDF (document format)",
    "PDM Systems": "",
    "PDM-Systeme": "",
    "PDMS": "",
    "PHP": "PHP (Programming Language)",
    "PL/SQL": "",
    "PLM - Product Livecycle Management": "",
    "PMI - Project Management Institut": "",
    "PMP - Project Management Professional": "",
    "POET": "",
    "PTC Creo": "",
    "PVCS": "",
    "Paediatrics": "",
    "Painter": "Painting",
    "Painter / Limer / Paperhanger": "Painting/Liming/Paperhaning",
    "Paralegal": "",
    "Pascal": "",
    "Patent Attorney": "",
    "Pathology": "",
    "Paver": "",
    "Payroll accounting": "",
    "Pedagogy": "",
    "Pediatric Surgery": "",
    "Peoplesoft": "",
    "Perl": "",
    "Pest Control": "",
    "Pharmaceutical Industry": "",
    "Pharmacology": "",
    "Pharmacy": "Pharmaceutical Expertise",
    "Pharmareferent": "",
    "Phlebology": "",
    "Photometrie": "",
    "Photovoltaics": "",
    "Physics": "",
    "Physiology": "",
    "Physiostherapy": "",
    "Pipeline Construction": "",
    "Planing and production systems": "",
    "Plant Engineering and Construction": "",
    "Playing Football": "",
    "Plumber": "Plumbing",
    "Pneumatic": "",
    "Pneumology": "",
    "Political Economics": "",
    "Political Science": "",
    "Poly Building": "",
    "Polymechanics": "",
    "Portlets": "",
    "Postfix": "",
    "Postgres SQL": "",
    "Power Station Construction": "",
    "PowerBuilder": "",
    "PowerPoint": "Microsoft PowerPoint",
    "Precision instruments engineering": "",
    "Prgress Modulus": "",
    "ProE": "",
    "ProTool": "",
    "Problem Management": "",
    "Process Control Systems": "",
    "Process Engineering": "",
    "Process Mechanics": "",
    "Process management": "",
    "Product finisher": "",
    "Production Engineering": "",
    "Production Mechanics": "",
    "Profibus DP/PA": "",
    "Programming Languages": "",
    "Prolog": "",
    "Psychiatry": "",
    "Psychology": "",
    "Public Relation (PR)": "Public relations (PR)",
    "Pump Engineering": "",
    "Python": "",
    "QML": "",
    "QlikView": "",
    "Qt": "",
    "QuarkXPress": "",
    "REXX": "",
    "RMI": "",
    "RPG": "",
    "RUP - Rational Unified Process": "",
    "Radar Technology": "",
    "Radio Engineering": "",
    "Radiology": "",
    "Radiotherapy": "",
    "Rational ClearCase": "",
    "Rational ClearQuest": "",
    "Rational Rose": "",
    "Re-insurance": "",
    "React": "",
    "Real Estate Law": "",
    "Recruiting Specialist": "",
    "RedDot": "",
    "Redis": "",
    "Regulatory Reporting": "",
    "Release Management": "",
    "Religious Studies": "",
    "Requirement Management": "",
    "Restaurant Management": "",
    "Retailer": "Retail",
    "Rheumatology": "",
    "Rhinoceros": "",
    "Riding": "",
    "Risk Management/Basel II/Basel III": "Risk Management",
    "Road Roller Driver": "",
    "Roadworks": "Roadworking",
    "Robotics": "",
    "Roller": "",
    "Roofer": "",
    "Ruby": "",
    "Ruplan": "",
    "SAN - System Area Network": "",
    "SAP - Industry Solutions": "",
    "SAP CRM": "",
    "SAP IS-H - Healthcaretive": "",
    "SAP IS-U - Utilities": "",
    "SAP Moduls": "SAP Modules",
    "SAP PDM": "",
    "SAP XI/PI - Process Integration": "",
    "SAP-BW - Business Information Warehouse": "",
    "SAP-CO - Controlling": "",
    "SAP-CS - Customer Service": "",
    "SAP-EHS - Environment/Health/Safety": "",
    "SAP-ERP - Enterprise Resource Planning": "",
    "SAP-FI - Finance": "",
    "SAP-HANA - High Performance Analytic Appliance": "",
    "SAP-HCM - Human Capital Management": "",
    "SAP-HR - Human Resources": "",
    "SAP-IM - Investment Management": "",
    "SAP-LE - Logistics Execution": "",
    "SAP-LO - Logistics General": "",
    "SAP-MM - Material Management": "",
    "SAP-NetWeaver": "",
    "SAP-PA - Personnel Administration General": "",
    "SAP-PLM - Product Lifecycle Mgmt": "",
    "SAP-PM - Plant Maintenance": "",
    "SAP-PP - Product Planning": "",
    "SAP-PS - Project Management": "",
    "SAP-QM - Quality Management": "",
    "SAP-R3": "SAP R/3",
    "SAP-RE - Real Estate Management": "",
    "SAP-SCM - Supply Chain Management (FSCM)": "",
    "SAP-SD - Sale & Distribution": "",
    "SAP-SEM -Strategic Enterprise Management": "",
    "SAP-SRM - Supplier Relationship Management": "",
    "SAP-WM - Warehouse Management": "",
    "SAS Software": "",
    "SCADA - Supervisory Control and Data Acquisition": "",
    "SCCM": "",
    "SEO/SEM - Search Engine Optimization & Marketing": "Search Engine Optimization (SEO)",
    "SESAM": "",
    "SMTP": "",
    "SNMP": "",
    "SOA - Service Oriented Architecture": "",
    "SOAP": "",
    "SQL": "SQL (Software)",
    "SQLite": "",
    "SSA (Baan)": "",
    "SWIFT": "",
    "SWT": "",
    "Sabre": "",
    "Saddler": "",
    "Sage Software": "",
    "Salesforce": "",
    "Scaffolding": "",
    "Scala": "",
    "Scientific knowledge": "",
    "Scrum": "",
    "Security / Cryptography": "",
    "Security Engineering": "",
    "Semantic Web": "",
    "Sempstress": "Seaming",
    "SensorTechnology": "",
    "Servlets": "",
    "Shiatsu": "",
    "Shipbuilding": "",
    "Siebel Systems": "",
    "Signal Processing": "",
    "Sitecore": "",
    "Six Sigma": "",
    "Skiing": "",
    "Skipper": "",
    "Smalltalk": "",
    "Smith": "",
    "Social Economy": "",
    "Social Insurance Law": "",
    "Social Law": "",
    "Social Pedagogy": "",
    "Social Science": "",
    "Soil Engineering": "",
    "SolidEdge": "",
    "SolidWorks": "SolidWorks (CAD Software)",
    "Sommelier": "",
    "Sonography": "",
    "Sony Vegas": "",
    "Sozilogy": "Sociology",
    "Specialist for bath operation": "",
    "Spectroscopy": "",
    "Spice": "",
    "Spinal Surgery": "",
    "Splunk": "",
    "Sport Management": "",
    "Sport Science": "",
    "Sports Skills": "",
    "Standards": "",
    "Steel Building": "",
    "Steel Building/Metal Building": "",
    "Step 7": "",
    "Stock management": "",
    "Stone Cutter": "",
    "Storage": "",
    "Stove-fitter": "",
    "Stucco Plasterer": "",
    "SubVersion": "",
    "SugarCRM": "",
    "Sun OS / Solaris": "",
    "Supply/Vendor management": "",
    "Surface Engineering": "",
    "Surgery": "",
    "Surgery Technique": "",
    "Surgery of the chest": "Thoracic Surgery",
    "Swift": "",
    "Swimming": "",
    "Swing": "",
    "Swiss-GAAP": "Swiss GAAP (accounting)",
    "Switching Technology": "",
    "Sybase": "",
    "Symphonie": "Symphonics",
    "Synthtical Engineering": "",
    "System G": "",
    "System Software": "",
    "Systems Mechanic": "",
    "TCM - Traditional Chinese Medicine": "TCM (Traditional Chinese Medicine)",
    "TCP/IP": "",
    "TIG Welder": "",
    "TOS": "",
    "TSO": "",
    "TYPO3": "TYPO3 (Software)",
    "Tailor": "Tailoring",
    "Tandem": "",
    "Tanner": "Tanning",
    "TargetLink": "",
    "Tattooer": "Tattooing",
    "Tax Law": "",
    "Tcl / Tcl/Tk": "",
    "TeX/LaTeX": "",
    "Team Foundation Server": "",
    "Teamcenter": "",
    "Technical Sales": "",
    "Telematics": "",
    "Tenancy Law": "",
    "Tennis": "",
    "Teradata": "",
    "Text Processing": "",
    "Textile Industry": "",
    "Textilindustrie": "Textile Industry",
    "Theology": "",
    "Tibco": "",
    "Tinline": "",
    "Titration": "",
    "Toolmaker": "Toolmaking",
    "Tourism": "",
    "Toxicology": "",
    "Track Construction": "",
    "Transactions in Securities": "",
    "Transformation Technique": "",
    "Transfusion Medicine": "",
    "Transmission Technology": "",
    "Transport Law": "",
    "Transportation/Shipping": "Transport",
    "Treasury": "Treasuring",
    "Tropical Medicine": "",
    "Tuina": "",
    "Tunnel Construction": "",
    "Turner": "",
    "UK-GAAP": "",
    "UML": "",
    "UMTS": "",
    "US-GAAP": "",
    "UX-Design": "",
    "Unigraphics NX": "",
    "Unit Testing": "",
    "Unix General": "Unix (Operating System)",
    "Unix SCO": "",
    "Unix Shell Script": "",
    "Urology": "",
    "V Model": "",
    "VB.NET": "",
    "VBA": "Microsoft Excel VBA",
    "VBScript": "",
    "VHDL": "",
    "VMware": "",
    "VPN": "",
    "Vascular Surgery": "",
    "Vault": "",
    "Veritas": "",
    "Veterinary Medical Professional": "",
    "Veterinary Medicine": "",
    "Vignette": "",
    "Vine Dresser": "",
    "Virology": "",
    "Virtualbox": "",
    "Visceral Surgery": "",
    "Visual Basic": "",
    "Visual FoxPro": "",
    "Visual Studio": "",
    "Vtiger": "",
    "VueJS": "",
    "WAN/LAN": "",
    "WCF/Indigo": "",
    "WML": "",
    "WPF/Avalon": "",
    "Waitering": "",
    "Weaver": "Weaving",
    "WebServices": "",
    "Welder": "Welding",
    "Wholesaler": "",
    "Wildfire": "",
    "Wildfly": "",
    "WinCC - Windows Control Center": "",
    "Windchill": "",
    "Windows": "Microsoft Windows",
    "Windows 10": "",
    "Windows 7": "",
    "Windows 95/98/Me": "",
    "Windows CE": "",
    "Windows NT": "",
    "Windows XP/2000/Vista": "",
    "Wire Erdoding": "EDM (Electrical Discharge Machining)",
    "Woodwork": "",
    "Woodworking": "",
    "Wordpress": "WordPress",
    "Workflow": "",
    "Worldspan": "",
    "X.25": "",
    "XML": "",
    "XML Parsing (SAX/DOM)": "",
    "XPath": "",
    "XSL/XSLT": "",
    "Xcode": "",
    "Zoology": "",
    "chimney sweep": "",
    "i-views": "",
    "iText": "",
    "keytech": "",
    "xBase / dBase": "",
    "z/OS": "",
    "Psychoanalysis": "",
    "Driving Licence C1 Light Trucks": "",
    "JBuilder": "",
    "Ingres": "",
    "Lotus Notes Script": "",
    "Bioprocess Engineering": "",
    "Information Elektronics": "",
    "Application Engineering": "",
    "SAP-MII - Manufactoring Integration": "",
    "Mambo": "",
    "Galenik": "",
    "NetBIOS": "",
    "Economic Philosophy": "",
    "High Frequency Engineering": "",
    "Zend Framework": "",
    "JAXB": "",
    "Misra-C": "",
    "Interbase": "",
    "Nuclear Medicine": "",
    "Easy2000": "",
    "JMX": "",
    "SAP-EC - Enterprise Controlling": "",
    "SAP-FI-AA - Asset Acconting": "",
    "Token Ring": "",
    "Economic Ethics": "",
    "ADT (Android Development Tools)": "",
    "FDDI": "",
    "Realtime Systems": "",
    "Graph Database": "",
    "Structural Mechanics": "",
    "PSpice": "",
    "Embedded C": "",
    "SAP IS-M - Media": "",
    "MaxDB (SAP DB)": "",
    "Splicer": "",
    "BPML/BPEL": "",
    "CruiseControl": "",
    "Refractormetry": "",
    "HP-UX": "",
    "MS BizTalk Server": "",
    "JNDI": "",
    "APQP (Advanced Product Quality Planning)": "",
    "Xamarin": "",
    "Holzmechaniker": "",
    "Flight Instructor": "",
    "OrCAD": "",
    "SAP-CO-PC - Product Costing": "",
    "SAP-PY Payroll": "",
    "Optoelectronics": "",
}


def get_skill_to_idx(a):
    mapping = get_mapping(a)
    return {k: i for i, k in enumerate(mapping.keys())}


def get_idx_to_skill(a):
    mapping = get_mapping(a)
    return {i: k for i, k in enumerate(mapping.keys())}


def get_mapping(a):
    # TODO ignore_skills, remove_synonyms
    return MAPPING
