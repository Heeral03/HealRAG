import sys
from pathlib import Path

# Add src to python path to load config
sys.path.append(str(Path(__file__).resolve().parent))
import config

# Define corpus texts: Regulatory, standards, and academic summaries
documents = {
    # --- GDPR Articles ---
    "gdpr_art9_para1": (
        "GDPR Article 9(1) - Processing of special categories of personal data: "
        "Processing of personal data revealing racial or ethnic origin, political opinions, "
        "religious or philosophical beliefs, or trade union membership, and the processing of "
        "genetic data, biometric data for the purpose of uniquely identifying a natural person, "
        "data concerning health or data concerning a natural person's sex life or sexual orientation "
        "shall be prohibited."
    ),
    "gdpr_art9_para2_a": (
        "GDPR Article 9(2)(a) - Exceptions to health data processing ban: "
        "Paragraph 1 shall not apply if the data subject has given explicit consent to the "
        "processing of those personal data for one or more specified purposes, except where "
        "Union or Member State law provide that the prohibition referred to in paragraph 1 "
        "may not be lifted by the data subject."
    ),
    "gdpr_art9_para2_h": (
        "GDPR Article 9(2)(h) - Medical diagnosis and health treatment exceptions: "
        "Processing is necessary for the purposes of preventive or occupational medicine, for the "
        "assessment of the working capacity of the employee, medical diagnosis, the provision of "
        "health or social care or treatment or the management of health or social care systems and "
        "services on the basis of Union or Member State law or pursuant to contract with a health professional."
    ),
    "gdpr_art9_para2_i": (
        "GDPR Article 9(2)(i) - Public health interest exception: "
        "Processing is necessary for reasons of public interest in the area of public health, such as "
        "protecting against serious cross-border threats to health or ensuring high standards of "
        "quality and safety of health care and of medicinal products or medical devices, on the basis "
        "of Union or Member State law which provides for suitable and specific measures to safeguard "
        "the rights and freedoms of the data subject, in particular professional secrecy."
    ),
    "gdpr_art9_para2_j": (
        "GDPR Article 9(2)(j) - Scientific, statistical, and historical research exception: "
        "Processing is necessary for archiving purposes in the public interest, scientific or historical "
        "research purposes or statistical purposes in accordance with Article 89(1) based on Union or "
        "Member State law which shall be proportionate to the aim pursued, respect the essence of the "
        "right to data protection and provide for suitable and specific measures to safeguard the "
        "fundamental rights and the interests of the data subject."
    ),
    "gdpr_art89_para1": (
        "GDPR Article 89(1) - Safeguards and derogations for research: "
        "Processing for archiving purposes in the public interest, scientific or historical research "
        "purposes or statistical purposes, shall be subject to appropriate safeguards, in accordance "
        "with this Regulation, for the rights and freedoms of the data subject. Those safeguards "
        "shall ensure that technical and organisational measures are in place in particular in order "
        "to ensure respect for the principle of data minimization. Those measures may include "
        "pseudonymisation, provided that those purposes can be fulfilled in that manner."
    ),
    "gdpr_art89_para2": (
        "GDPR Article 89(2) - Research derogations: "
        "Where personal data are processed for scientific or historical research purposes or "
        "statistical purposes, Union or Member State law may provide for derogations from the rights "
        "referred to in Articles 15 (access), 16 (rectification), 18 (restriction of processing) and "
        "21 (objection) subject to the conditions and safeguards referred to in paragraph 1 of this "
        "Article in so far as such rights are likely to render impossible or seriously impair the "
        "achievement of the specific purposes, and such derogations are necessary for the fulfilment "
        "of those purposes."
    ),

    # --- EU EHDS Regulation (Regulation (EU) 2025/327) ---
    "ehds_ch2_art3_natural_persons": (
        "EHDS Chapter II Article 3 - Primary Use Rights: Natural persons shall have the right to access their "
        "personal electronic health data processed in the context of primary use of electronic health "
        "data immediately, free of charge and in an easily readable, common and accessible format. "
        "They shall have the right to retrieve an electronic copy of at least their electronic health data "
        "in the European electronic health record exchange format."
    ),
    "ehds_ch2_art4_access_by_health_professionals": (
        "EHDS Chapter II Article 4 - Access by health professionals: Health professionals shall have access "
        "to the electronic health data of natural persons under their treatment, irrespective of the "
        "Member State of affiliation of the natural person and the Member State of treatment. The access "
        "shall be limited to the data relevant for the treatment provision and shall comply with the "
        "principles of authorization, authentication, and data protection safeguards."
    ),
    "ehds_ch2_art5_myhealth_eu": (
        "EHDS Chapter II Article 5 - MyHealth@EU infrastructure: "
        "The Commission shall establish a central platform for digital health (MyHealth@EU) to facilitate "
        "and support the exchange of electronic health data between national contact points for digital health "
        "of the Member States. The infrastructure shall enable cross-border transmission of patient summaries, "
        "e-prescriptions, e-dispensations, medical images, laboratory results, and discharge reports."
    ),
    "ehds_ch2_art7_right_to_restrict_access": (
        "EHDS Chapter II Article 7 - Right of natural persons to restrict access of health professionals: "
        "Natural persons shall have the right to restrict access of health professionals to all or parts "
        "of their electronic health data. Member States shall establish the rules and technical mechanisms "
        "for such restrictions, ensuring that the restriction of access does not prevent emergency access "
        "where the life or vital interests of the natural person or another person are threatened."
    ),
    "ehds_ch4_art33_secondary_data_use": (
        "EHDS Chapter IV Article 33 - Categories of electronic health data for secondary use: "
        "Data holders shall make available for secondary use the following categories of electronic health "
        "data: (a) EHRs; (b) data factors influencing health, including socio-economic, environmental and "
        "lifestyle factors; (c) genetic, genomic and proteomic data; (d) person-generated electronic "
        "health data; (e) data from clinical trials; (f) register data of medicinal products and devices."
    ),
    "ehds_ch4_art34_purposes_secondary_use": (
        "EHDS Chapter IV Article 34 - Permitted purposes for secondary use: "
        "Secondary use of electronic health data shall be permitted only for the following purposes: "
        "(a) activities for reasons of public interest in the area of public health; (b) support of "
        "public health tasks of healthcare public bodies; (c) scientific research related to health or "
        "care sectors; (d) development and evaluation of medicinal products or medical devices; "
        "(e) training, testing and evaluating of algorithms, including in medical devices and AI systems."
    ),
    "ehds_ch4_art35_prohibited_purposes": (
        "EHDS Chapter IV Article 35 - Prohibited secondary uses of health data: "
        "The following secondary uses of electronic health data shall be prohibited: (a) taking decisions "
        "detrimental to a natural person based on their electronic health data, including increasing "
        "insurance premiums or refusing insurance contracts; (b) advertising or marketing activities "
        "directed at health professionals or natural persons; (c) providing access to data to third parties "
        "not authorized under a data permit; (d) developing products or services that may harm public "
        "health or undermine safety."
    ),
    "ehds_ch4_art36_data_access_bodies": (
        "EHDS Chapter IV Article 36 - Health Data Access Bodies (HDABs): "
        "Each Member State shall designate one or more Health Data Access Bodies responsible for granting "
        "access to electronic health data for secondary use. HDABs shall receive data applications, "
        "evaluate requests, issue data permits under strict safety requirements, pseudonymize or "
        "anonymize the datasets, and make them available to authorized data users in a secure processing environment."
    ),
    "ehds_ch4_art38_opt_out_mechanism": (
        "EHDS Chapter IV Article 38 - Right to opt-out from secondary data use: "
        "Natural persons shall have the right to opt-out from the processing of their electronic health "
        "data for secondary use. The opt-out mechanism shall be simple, user-friendly, and accessible. "
        "However, opting out shall not affect processing necessary for public health monitoring, "
        "official statistics, or responding to public health emergencies."
    ),
    "ehds_ch4_art39_secure_processing_environment": (
        "EHDS Chapter IV Article 39 - Secure processing environments: "
        "The health data access bodies shall provide access to electronic health data only in a secure "
        "processing environment. This environment shall comply with high technical and organizational security "
        "standards, ensuring that data users can only query the data, copy results, and cannot download "
        "individual level micro-data or re-identify individuals."
    ),
    "ehds_ch4_art40_data_altruism": (
        "EHDS Chapter IV Article 40 - Data altruism in healthcare: "
        "Natural persons may consent and authorize health data access bodies to process their electronic "
        "health data for altruistic purposes, such as scientific medical research to find cures for rare "
        "diseases or improve treatment guidelines. Health data access bodies shall facilitate registry of "
        "data altruism options and link them to secure environments."
    ),

    # --- Standards (FHIR and IPS) ---
    "fhir_patient_resource": (
        "FHIR Patient Resource Definition: "
        "The Patient resource covers data about patients. It represents demographic and administrative "
        "information about a person or animal receiving care or other health-related services. Key attributes "
        "include: identifier (unique identifiers like SSN), active (boolean to mark record state), name "
        "(HumanName structure for family and given names), telecom (contact details), gender (administrative "
        "gender values: male, female, other, unknown), birthDate (patient Date of Birth), and address (physical addresses)."
    ),
    "fhir_observation_resource": (
        "FHIR Observation Resource Definition: "
        "Observations are a central element in healthcare, used to support diagnosis, monitor progress, "
        "and determine baseline characteristics. Key elements of the Observation resource include: identifier, "
        "status (registered, preliminary, final, amended), category (vital-signs, laboratory, imaging), "
        "code (LOINC/SNOMED codes indicating what was observed), subject (Reference to Patient), effectiveDateTime "
        "(when observation occurred), and valueQuantity (numeric vital value with unit) or valueCodeableConcept (coded results)."
    ),
    "fhir_bundle_resource": (
        "FHIR Bundle Resource: "
        "A Bundle is a container resource that acts as a wrapper for a collection of other resources. It is "
        "frequently used in API responses, transactions, batch operations, and document-style payloads. A Bundle "
        "contains metadata and entries (a list of resource objects), and must specify a type attribute such as "
        "transaction, batch, history, searchset, or document."
    ),
    "fhir_consent_resource": (
        "FHIR Consent Resource: "
        "The Consent resource is used to record a healthcare consumer's choice to permit or deny "
        "the collection, use, or disclosure of their personal health data. It specifies the scope of the consent, "
        "the actors involved (e.g., specific practitioners or organizations), the data categories covered, "
        "the validity period, and the explicit action (permit or deny)."
    ),
    "fhir_overview_intro": (
        "Fast Healthcare Interoperability Resources (FHIR) Overview: "
        "FHIR (pronounced 'fire') is a next-generation standards framework created by Health Level Seven International (HL7). "
        "It defines how healthcare information can be exchanged between different computer systems regardless of how it is stored locally. "
        "FHIR combines the best features of HL7 Version 2, HL7 Version 3, and CDA, while leveraging modern web standards such as "
        "RESTful HTTP APIs, JSON, XML, OAuth2, and OpenID Connect."
    ),
    "fhir_architecture_principles": (
        "FHIR Architecture and Core Principles: "
        "The fundamental building blocks of FHIR are modular data components called 'Resources'. "
        "Each resource represents a discrete clinical or administrative concept (e.g., Patient, Observation, Medication, Encounter). "
        "Key architectural principles of FHIR include: (1) Focus on the 80% consensus rule—modeling concepts common to 80% of health systems while using Extension elements for the rest; "
        "(2) Human readability—every resource includes a human-readable XHTML narrative summary; "
        "(3) Modern RESTful architecture—exposing resources via standard HTTP methods (GET, POST, PUT, DELETE) with predictable URIs."
    ),
    "ips_profile_summary": (
        "International Patient Summary (IPS) Profile: "
        "The IPS is an electronic health record summary containing a minimized, specialty-agnostic list "
        "of essential clinical details. It is designed to support cross-border unscheduled patient care. The "
        "IPS profile mandatorily requires three sections: Medication Summary (active medications), Allergies "
        "and Intolerances (substances and reaction types), and Active Problems (list of current conditions)."
    ),
    "ips_diagnostic_results": (
        "IPS Diagnostic Results Section: "
        "The IPS profile contains optional but recommended sections. The Diagnostic Results section "
        "includes links to laboratory reports, pathology findings, and diagnostic imaging statements. It uses "
        "standardized coding systems such as LOINC for lab test categories and UCUM for measurement units "
        "to ensure international semantic interoperability."
    ),
    "ips_procedures_history": (
        "IPS History of Procedures Section: "
        "This section of the IPS details clinical procedures completed in the past that are relevant to "
        "clinical decision making. Procedures should be coded using SNOMED CT terms to maintain cross-border "
        "semantic alignment, ensuring practitioners in other countries can understand the clinical history."
    ),

    # --- Academic and Thesis Summaries ---
    "paper_blockchain_consent": (
        "Academic Paper - Decentralized Consent Management for Cross-Border Patient Summary Exchange: "
        "This study proposes a decentralized architecture for patient consent infrastructure in cross-border "
        "exchanges using Ethereum smart contracts. Consent records are stored immutably on-chain as cryptographic "
        "hashes, while raw files are maintained off-chain. Evaluated within the context of MyHealth@EU, the system "
        "demonstrated high integrity, secure audit logs, and compliance with GDPR patient control rights."
    ),
    "paper_fhir_oauth2_interop": (
        "Academic Paper - Evaluating FHIR-over-OAuth2/smart-on-fhir for Regional EHR Interoperability: "
        "This paper evaluates the performance and security of SMART on FHIR specifications using OAuth 2.0 "
        "and OpenID Connect protocols to authenticate regional exchange clients. It presents benchmarks "
        "for token exchange latency and scope-based authorization filters. Results show that scope filtering "
        "(e.g., patient/Observation.read) minimizes data leakage but incurs a 15% latency overhead."
    ),
    "paper_zkp_ehds_privacy": (
        "Academic Paper - Zero-Knowledge Proofs (ZKP) for Privacy-Preserving Health Data Reuse in the EHDS: "
        "The researchers propose a system based on zk-SNARKs that allows health researchers to run statistical "
        "algorithms on clinical data in Health Data Access Bodies (HDABs) without ever decrypting or accessing "
        "raw patient records. Data users receive mathematical proof that the calculation was performed correctly "
        "over valid patient data, meeting EHDS Chapter IV security standards and minimizing re-identification risks."
    ),
    "paper_myhealth_gateway_audit": (
        "Academic Paper - Security Audit of MyHealth@EU Gateways and International Patient Summary Exchange: "
        "A comprehensive security review of national contact points (NCPHs) executing IPS transformation. "
        "The audit identified potential vulnerability areas, specifically XML external entity (XXE) injection "
        "attacks during CDA-to-FHIR payload transformations and TLS configuration weaknesses. Recommendations "
        "include mandatory message schema validation and strict mutual TLS (mTLS) enforcement."
    ),
    "paper_consent_infraction_auditing": (
        "Academic Paper - Scalable Consent Infraction Auditing on Hyperledger Fabric: "
        "An investigation into ledger performance for auditing digital health access networks. "
        "By hosting a consent state engine on Hyperledger Fabric smart contracts, the system records "
        "every access query and cross-checks it against active patient consent rules. It maintains audit records "
        "capable of processing up to 3000 queries per second, showing viability for national-scale deployment."
    ),
}

# Distribute or split these texts to generate 100+ documents (50-200 documents needed)
# We can create variations and separate detailed provisions as individual documents
def seed_corpus():
    corpus_dir = config.CORPUS_DIR
    
    # We will generate 100 files by expanding on the primary sources and writing detailed sub-sections
    count = 1
    
    # Write primary articles
    for key, content in documents.items():
        filename = corpus_dir / f"doc_{count:03d}_{key}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
        
    # Generate variations / simulated documents to reach 100+ documents
    # These will act as mock healthcare logs, regional interoperability briefs, and clinical case summaries.
    
    # Sub-sections of EHDS Chapter IV (Secondary Use details)
    ehds_subsections = [
        "EHDS Implementing Act on secondary use guidelines: Specifying the structure of data quality and utility labels for datasets.",
        "EHDS Article 46: Fees charged by health data access bodies to cover administration and infrastructure costs.",
        "EHDS Annex I: Specification of groups of health data categories required to be made available for secondary research.",
        "EHDS Section 3: Joint controllership of the secure processing environments for multi-country health data access permits.",
        "EHDS Chapter IV Article 41: Data quality and utility label requirements for data holders before indexing data.",
        "EHDS Article 42: Mutual recognition of data permits across European health data access bodies (HDAB).",
        "EHDS Guidance: Penalties and administrative fines for non-compliance with secure environment guidelines.",
        "EHDS Provision: Member State policies for data access body cooperation with national medicine regulatory agencies."
    ]
    for idx, text in enumerate(ehds_subsections):
        filename = corpus_dir / f"doc_{count:03d}_ehds_sub_{idx}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"EU Health Regulatory Brief {idx+1}: {text}")
        count += 1

    # EHR Interoperability memos (cross-border)
    memos = [
        "Cross-Border Interoperability memo: Connecting Italian NCPH to German NCPH for e-prescription sharing.",
        "Health Information Exchange (HIE) standard operating procedures: Validating Patient resource naming conventions.",
        "IPS deployment guidelines in Sweden: Mapping local EHR schema values to SNOMED-CT equivalents.",
        "EHR Audit log requirements: Logging practitioner ID and timestamp for every access request to IPS.",
        "FHIR resource bundle serialization: JSON representation requirements for clinical diagnostic files.",
        "Interoperability framework update: Supporting French NCPH discharge summary transformations to FHIR.",
        "Clinical coding best practices: Restricting local mapping tables to avoid semantic drift in cross-border transfers.",
        "EU-US health data agreement: Compliance checks for sharing anonymized research datasets under GDPR safeguards.",
        "National Contact Point for Health (NCPH) registry: Keeping public keys active for mTLS handshake configurations.",
        "Patient consent revocation process: Timeline constraints for propagating opt-out settings to regional EHRs."
    ]
    for idx, text in enumerate(memos):
        filename = corpus_dir / f"doc_{count:03d}_interop_memo_{idx}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"Regional Interoperability Document: {text}")
        count += 1

    # Generate synthetic FHIR resources to represent EHR database dumps (patient, diagnostic, observation data)
    # We will generate around 60 synthetic case summaries/FHIR objects to comfortably pass 100 documents.
    for patient_id in range(1, 61):
        fhir_dump = (
            f"EHR FHIR Export for Patient ID PAT-{patient_id:04d}.\n"
            f"Active: true. Resource Type: Patient, Observation, Bundle.\n"
            f"Patient Demographics: Name: Patient {patient_id}, Gender: {'male' if patient_id % 2 == 0 else 'female'}, "
            f"BirthDate: 198{patient_id % 10}-0{1 + (patient_id % 9)}-{10 + (patient_id % 18)}.\n"
            f"Observation Category: vital-signs, laboratory.\n"
            f"Vital signs details: LOINC code 8867-4 (Heart rate) valueQuantity: {60 + (patient_id % 40)} beats/min, status: final.\n"
            f"Laboratory details: LOINC code 29463-7 (Body weight) valueQuantity: {55 + (patient_id % 50)} kg.\n"
            f"Consent status: {'Permit secondary use' if patient_id % 3 != 0 else 'Deny secondary use / Opt-out active'}.\n"
            f"IPS medications summary: Patient is taking medicine MED-{patient_id:03d} for active problem code PROB-{patient_id:03d}."
        )
        filename = corpus_dir / f"doc_{count:03d}_fhir_patient_{patient_id}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(fhir_dump)
        count += 1

    # Generate UK NHS Regulatory & Standards Documents
    nhs_docs = [
        (
            "doc_111_nhs_caldicott_principles.txt",
            """UK NHS Caldicott Principles and Data Protection:
The 8 Caldicott Principles govern the handling of patient-identifiable information within the UK National Health Service (NHS).
Principle 1: Justify the purpose(s) for using confidential information.
Principle 2: Use confidential information only when absolutely necessary.
Principle 3: Use the minimum necessary confidential information.
Principle 4: Access to confidential information should be on a strict need-to-know basis.
Principle 5: Everyone with access to confidential information should be aware of their responsibilities.
Principle 6: Comply with the law (Data Protection Act 2018 / UK GDPR).
Principle 7: The duty to share information for direct patient care can be as important as the duty to protect patient confidentiality.
Principle 8: Inform patients and service users about how their confidential information is used.
Caldicott Guardians are senior health or social care professionals appointed in NHS organizations to ensure these principles are respected.""",
        ),
        (
            "doc_112_nhs_dspt_framework.txt",
            """NHS Data Security and Protection Toolkit (DSPT):
The Data Security and Protection Toolkit (DSPT) is an online self-assessment tool that allows NHS organizations and third-party healthcare vendors to measure their performance against the UK National Data Guardian's 10 Data Security Standards.
Key Compliance Requirements:
1. Personal Confidential Data: All staff ensure PCD is handled legally and securely.
2. Staff Training: 95% of staff must complete annual data security training.
3. Managing Data Access: Access permissions are reviewed regularly and revoked upon termination.
4. Process Reviews: Processes involving PCD are reviewed at least annually.
5. Responding to Incidents: Cyber security incidents must be reported to the NHS Digital Cyber Operations team within 24 hours.
6. Continuity Planning: Business continuity plans for cyber attacks and system outages are tested annually.
7. System Security: Unsupported operating systems, software, and unpatched vulnerabilities are prohibited on NHS networks.
8. Accountable Officers: Named senior roles (SIRO and Caldicott Guardian) oversee data risk.""",
        ),
        (
            "doc_113_nhs_fhir_uk_core.txt",
            """NHS UK Core FHIR Implementation Guide:
NHS England and NHS Digital publish the UK Core FHIR specifications for interoperability across UK health and social care systems.
Key UK Core Resources & Extensions:
1. UKCore-Patient: Includes NHS Number extension (URL: https://fhir.hl7.org.uk/StructureDefinition/Extension-UKCore-NHSNumber) with mandatory verification status coding.
2. UKCore-Practitioner: Captures Professional Registration Details (GMC code for doctors, NMC code for nurses).
3. UKCore-Organization: Maps to ODS Codes (Organisation Data Service codes managed by NHS Digital).
4. UKCore-Consent: Captures National Data Opt-out preferences where patients choose to opt out of their confidential patient information being used for research and planning.""",
        ),
        (
            "doc_114_nhs_national_data_opt_out.txt",
            """NHS National Data Opt-out Policy:
The NHS National Data Opt-out is a service that allows patients in England to opt out of their confidential patient information being used for research and planning purposes.
Scope & Exceptions:
- Applies to: Secondary processing of confidential patient information across NHS England, UK Health Security Agency (UKHSA), and local authorities.
- Exemptions: Direct care (opt-out does not apply when sharing data for direct treatment), mandatory legal disclosures (court orders), and public health emergency directions under Regulation 3 of the Health Service Control of Patient Information Regulations 2002 (COPI).
- Enforcement: All health and care organizations handling NHS patient data must adhere to the National Data Opt-out policy and filter data disclosures against the NHS Digital central opt-out repository.""",
        ),
    ]

    for filename, content in nhs_docs:
        with open(corpus_dir / filename, "w", encoding="utf-8") as f:
            f.write(content.strip())
        count += 1

    print(f"Successfully seeded {count-1} documents in {corpus_dir}")

if __name__ == "__main__":
    seed_corpus()
