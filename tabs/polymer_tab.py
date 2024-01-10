# tab2.py
import streamlit as st

def display(tab):
    # Section: What are Polymers?
    tab.header("What are Polymers?")
    tab.write(
        """
        Polymers are chemical substances composed of macromolecules. These macromolecules are constructed from one or more 
        structural units known as constitutional repeating units. The term 'polymer' is derived from the Greek words for 
        'many parts,' reflecting how polymers are built up from many repeated units. Polymers can vary significantly, with 
        different molecular masses and repeating unit counts.
        """
    )

    # Section: Types of Polymers
    tab.header("Types of Polymers")
    tab.subheader("Homopolymers")
    tab.write(
        """
        Homopolymers consist of a single type of monomer.Natural rubber is a type of polyisoprene and is an example of a
        natural homopolymer. Additionally, some of the most common homopolymers are, for instance, polyethylene,
        known for its use in everything from plastic bags to bottles; polypropylene, which is utilized in various
        applications ranging from packaging to automotive parts; and polystyrene, often seen in packaging, insulation,
        and disposable food ware.
        """
    )
    hcol1,hcol2,hcol3 = tab.columns(3)
    hcol1.write("Polyethylen (PE)")
    hcol1.image("pictures/polyethylen_final.png")
    hcol2.write("Polypropylen (PP)")
    hcol2.image("pictures/polypropylen_final.png")
    hcol3.write("Polystyrol (PS)")
    hcol3.image("pictures/polystyrol_final.png")

    #polymere_pic_col.image("pictures/homo_poly.png")

    tab.subheader("Copolymers")
    tab.write(
        """
        Copolymers are polymers that are composed of at least two different types of monomers,
        which gives them a combination of properties from each monomer unit. They can exhibit characteristics that are
        not found in homopolymers and can be designed to have specific mechanical, thermal, or chemical properties.
        Examples include Acrylonitrile Butadiene Styrene (ABS), 
        Styrene-Acrylonitrile (SAN), and butyl rubber(IIR). Most biopolymers are copolymers.
        """
    )

    ccol1,ccol2,ccol3 = tab.columns(3)
    ccol2.image("pictures/Styrene-acrylonitrile.png")

    tab.subheader("Polymer Blends")
    tab.write(
        """
        Polymer blends are produced by mixing different homopolymers and/or copolymers. They are usually created through 
        the intensive mechanical mixing of molten polymers, resulting in a homogeneous material. A special type of blend, 
        known as a polymer alloy, is also included in this category.
        Here are two examples:
        
        1.ABS/PC Blends: These are used in the automotive and electronics industries for parts like bumpers and computer
         cases because they offer a good balance of toughness and heat resistance.

        2.PVC/NBR Blends: Applied in making oil-resistant industrial hoses and gaskets due to their enhanced chemical resistance and flexibility.
        """
    )

    # Section: Polymer Structure
    tab.header("Polymer Structure")
    tab.write(
        """
        The macromolecules formed during synthesis have different basic structures that determine the physical properties 
        of the polymer. These structures can be linear, branched, or networked, affecting characteristics like density, 
        strength, and melting point. The degree of branching influences how the molecules interact within the solid polymer, 
        with highly branched polymers being amorphous and unbranched ones forming semi-crystalline structures.
        """
    )
    tab.image("pictures/polymere_struct.png")

    # Section: Conductive Polymers
    tab.header("Conductive Polymers")
    tab.write(
        """
        Conductive polymers require the presence of conjugated pi-electron systems for electrical conductivity. However, 
        they initially remain insulators or at best, semiconductors. Conductivity comparable to metallic conductors only 
        occurs when polymers are doped oxidatively or reductively, enhancing conductivity with increasing crystallinity.
        """
    )

    tab.header("Leveraging Polymers in Machine Learning: SMILES Strings and Their Fingerprinting")
    tab.write("""
    In my project, I harness the power of polymers in the realm of machine learning. To achieve this, I employ SMILES strings,
    a chemical notation system, and convert them into unique fingerprints that are instrumental for machine learning applications.
    """)
    tab.subheader("Understanding SMILES Strings")
    tab.markdown("""
    **SMILES** (Simplified Molecular Input Line Entry System) is a concise and human-readable representation of chemical compounds,
     especially organic molecules. It uses a string of characters to encode the structural information of a molecule,
      making it a versatile choice for conveying complex molecular structures in a simple, standardized format.

    SMILES strings consist of elemental symbols, numeric values, and special characters. They describe atoms,
    bonds, and connectivity, allowing chemists and researchers to communicate molecular structures easily.
     For example, the SMILES string for water is "O" for oxygen and "H" for hydrogen, represented as "O-H."
    """)

    sscol1,sscol2 = tab.columns(2)
    sscol1.code("""
    from psmiles import PolymerSmiles as PS
    # polyethylene
    polyethylene = PS("[*]CC[*]")
    polyethylene
    """)
    sscol2.image("pictures/polyethylenPS.png")
    tab.subheader("SMILES String Fingerprinting")
    tab.write("""
    In the context of machine learning, one of the key challenges is to quantify and compare the structural features of
    different molecules. This is where SMILES fingerprinting comes into play. SMILES fingerprints are a means of
    representing the structural characteristics of a molecule in a binary or numerical format. These fingerprints are
    designed to capture essential structural information while reducing the complexity of the molecule.
    
    For instance, I convert SMILES strings from polymer structures into unique
    fingerprints using the [psmiles Python package](https://github.com/Ramprasad-Group/psmiles).
    The Morgan fingerprint captures the topological features of a molecule by identifying
    atom neighborhoods. These fingerprints are binary vectors where each bit represents the presence or absence of a
    specific structural motif in the molecule.
    
    Using SMILES fingerprints, I can compare polymer structures, identify similarities, and apply machine learning
    techniques to predict properties or behavior. Whether it's classifying polymers, predicting their properties,
    or optimizing chemical reactions, SMILES fingerprinting is a valuable tool in the machine learning arsenal for
    polymer-related projects.
    
    """)
    # Source
    tab.subheader("Source")
    tab.write(
        """
        [Polymer - Wikipedia](https://de.wikipedia.org/wiki/Polymer)
        """
    )



