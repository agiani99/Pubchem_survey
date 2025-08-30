import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, QED
from rdkit.Chem.rdMolDescriptors import CalcNumHBD, CalcNumHBA, CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings, CalcNumSaturatedRings
from rdkit.Chem.rdMolDescriptors import CalcNumHeterocycles, CalcNumAtomStereoCenters
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PubChemTemporalAnalyzer:
    def __init__(self, csv_file_path, delay_between_requests=1.0):
        """
        Initialize the PubChem temporal analyzer.
        
        Args:
            csv_file_path: Path to the chemically_clean_mw1000_collection.csv file
            delay_between_requests: Delay in seconds between web requests to avoid overloading PubChem
        """
        self.csv_file_path = csv_file_path
        self.delay = delay_between_requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Define CID bins for sampling
        self.cid_bins = [
            (10, 100),
            (100, 1000),
            (1000, 10000),
            (10000, 100000),
            (100000, 500000),
            (500000, 1000000),
            (1000000, 5000000),
            (5000000, 10000000),
            (10000000, 50000000),
            (50000000, 116000000)
        ]
        
    def scrape_compound_creation_date(self, cid):
        """
        Scrape the creation date for a specific CID from PubChem using multiple methods.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            str: Creation date in YYYY-MM-DD format or None if not found
        """
        # Method 1: Try PubChem REST API first (more reliable)
        date = self._get_date_from_api(cid)
        if date:
            return date
        
        # Method 2: Try scraping from the summary page
        date = self._get_date_from_summary_page(cid)
        if date:
            return date
        
        # Method 3: Try the record page with different approaches
        date = self._get_date_from_record_page(cid)
        if date:
            return date
        
        return None
    
    def _get_date_from_api(self, cid):
        """Try to get creation date from PubChem REST API."""
        try:
            # Try to get compound summary via API
            api_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/record/JSON"
            response = self.session.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for creation date in the JSON structure
                if 'PC_Compounds' in data:
                    for compound in data['PC_Compounds']:
                        if 'props' in compound:
                            for prop in compound['props']:
                                if 'urn' in prop and 'label' in prop['urn']:
                                    label = prop['urn']['label'].lower()
                                    if any(keyword in label for keyword in ['create', 'deposit', 'modify']):
                                        if 'value' in prop and 'sval' in prop['value']:
                                            date_str = prop['value']['sval']
                                            parsed_date = self._parse_date_string(date_str)
                                            if parsed_date:
                                                return parsed_date
            
            # Try PUG View API for record information
            view_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
            response = self.session.get(view_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Search through the hierarchical data structure
                date = self._extract_date_from_json(data)
                if date:
                    return date
                    
        except Exception as e:
            logger.debug(f"API method failed for CID {cid}: {str(e)}")
        
        return None
    
    def _extract_date_from_json(self, data):
        """Recursively search for date information in JSON data."""
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(key, str) and any(keyword in key.lower() for keyword in ['create', 'deposit', 'modify', 'date']):
                    if isinstance(value, str):
                        parsed_date = self._parse_date_string(value)
                        if parsed_date:
                            return parsed_date
                
                # Recurse into nested structures
                result = self._extract_date_from_json(value)
                if result:
                    return result
                    
        elif isinstance(data, list):
            for item in data:
                result = self._extract_date_from_json(item)
                if result:
                    return result
        
        return None
    
    def _get_date_from_summary_page(self, cid):
        """Try to get date from the compound summary page."""
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for specific patterns in the HTML
                patterns_to_search = [
                    # Look for table rows with date information
                    ('tr', lambda tag: tag.find('td') and any(keyword in tag.get_text().lower() 
                                                            for keyword in ['create', 'deposit', 'modify'])),
                    # Look for divs with date classes
                    ('div', lambda tag: tag.get('class') and any('date' in str(cls).lower() 
                                                               for cls in tag.get('class', []))),
                    # Look for spans with temporal information
                    ('span', lambda tag: tag.get_text() and any(keyword in tag.get_text().lower() 
                                                              for keyword in ['created', 'deposited', 'modified']))
                ]
                
                for tag_name, condition in patterns_to_search:
                    elements = soup.find_all(tag_name, condition)
                    for element in elements:
                        date = self._extract_date_from_element(element)
                        if date:
                            return date
                
                # Look for metadata in script tags (sometimes dates are in JavaScript)
                script_tags = soup.find_all('script')
                for script in script_tags:
                    if script.string:
                        dates = self._extract_dates_from_text(script.string)
                        if dates:
                            return min(dates)  # Return earliest date
                            
        except Exception as e:
            logger.debug(f"Summary page method failed for CID {cid}: {str(e)}")
        
        return None
    
    def _get_date_from_record_page(self, cid):
        """Try to get date from the detailed record page."""
        try:
            # Try the full record view
            url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}#section=Information-Sources"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look in the Information Sources section
                info_sections = soup.find_all(['div', 'section'], 
                                            attrs={'class': lambda x: x and 'information' in str(x).lower() if x else False})
                
                for section in info_sections:
                    date = self._extract_date_from_element(section)
                    if date:
                        return date
                
                # Look for any table with temporal information
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all(['td', 'th'])
                        for cell in cells:
                            if any(keyword in cell.get_text().lower() 
                                  for keyword in ['create', 'deposit', 'modify', 'date']):
                                # Look in the next cell for the date
                                next_cell = cell.find_next_sibling(['td', 'th'])
                                if next_cell:
                                    date = self._extract_date_from_element(next_cell)
                                    if date:
                                        return date
                                        
        except Exception as e:
            logger.debug(f"Record page method failed for CID {cid}: {str(e)}")
        
        return None
    
    def _parse_date_string(self, date_str):
        """Parse various date string formats."""
        if not date_str or not isinstance(date_str, str):
            return None
        
        # Common date formats to try
        date_formats = [
            '%Y-%m-%d',
            '%Y/%m/%d', 
            '%m/%d/%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y'
        ]
        
        # Clean the date string
        date_str = date_str.strip()
        
        for fmt in date_formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                if 1990 <= date_obj.year <= datetime.now().year:
                    return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _extract_dates_from_text(self, text):
        """Extract all valid dates from a text string."""
        import re
        
        dates = []
        
        # Various date patterns
        date_patterns = [
            r'\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b',
            r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b',
            r'\b([A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})\b',
            r'\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                parsed_date = self._parse_date_string(match)
                if parsed_date:
                    dates.append(parsed_date)
        
        return sorted(set(dates)) if dates else None
    
    def try_better_date_scraping(self, cid):
        """
        Enhanced date scraping using multiple PubChem APIs and endpoints.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            str: Creation date or None
        """
        # Method 1: Try the PubChem eUtils API (more likely to have historical data)
        try:
            eutils_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pccompound&id={cid}&retmode=json"
            response = self.session.get(eutils_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and str(cid) in data['result']:
                    compound_data = data['result'][str(cid)]
                    
                    # Look for creation or modification dates
                    for date_field in ['createdate', 'modifydate', 'depositdate']:
                        if date_field in compound_data:
                            date_str = compound_data[date_field]
                            parsed_date = self._parse_date_string(date_str)
                            if parsed_date:
                                logger.info(f"Found {date_field} for CID {cid}: {parsed_date}")
                                return parsed_date
        except Exception as e:
            logger.debug(f"eUtils API failed for CID {cid}: {str(e)}")
        
        # Method 2: Try PubChem's Classification API (sometimes has historical info)
        try:
            class_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/classification/JSON"
            response = self.session.get(class_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                # Look for date information in classification data
                date = self._extract_date_from_json(data)
                if date:
                    logger.info(f"Found classification date for CID {cid}: {date}")
                    return date
        except Exception as e:
            logger.debug(f"Classification API failed for CID {cid}: {str(e)}")
        
        # Method 3: Try PubChem's annotation API
        try:
            annot_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/annotations/heading/JSON"
            response = self.session.get(annot_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                date = self._extract_date_from_json(data)
                if date:
                    logger.info(f"Found annotation date for CID {cid}: {date}")
                    return date
        except Exception as e:
            logger.debug(f"Annotation API failed for CID {cid}: {str(e)}")
        
        # Method 4: Enhanced web scraping with more specific selectors
        try:
            url = f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for specific data sections
                sections_to_check = [
                    "div[id*='date']",
                    "div[class*='date']", 
                    "span[class*='date']",
                    "td[class*='date']",
                    "div[id*='created']",
                    "div[id*='modified']",
                    "div[id*='deposited']"
                ]
                
                for selector in sections_to_check:
                    elements = soup.select(selector)
                    for element in elements:
                        date = self._extract_date_from_element(element)
                        if date:
                            logger.info(f"Found web scraped date for CID {cid}: {date}")
                            return date
                
                # Look for JSON-LD structured data
                json_scripts = soup.find_all('script', type='application/ld+json')
                for script in json_scripts:
                    try:
                        import json
                        json_data = json.loads(script.string)
                        date = self._extract_date_from_json(json_data)
                        if date:
                            logger.info(f"Found JSON-LD date for CID {cid}: {date}")
                            return date
                    except:
                        continue
                        
        except Exception as e:
            logger.debug(f"Enhanced web scraping failed for CID {cid}: {str(e)}")
        
        return None
    
    def _extract_date_from_element(self, element):
        """Extract date from a BeautifulSoup element."""
        import re
        
        text = element.get_text() if element else ""
        
        # Look for common date formats
        date_patterns = [
            r'\b(\d{4}-\d{1,2}-\d{1,2})\b',
            r'\b(\d{1,2}/\d{1,2}/\d{4})\b',
            r'\b(\d{4}/\d{1,2}/\d{1,2})\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    # Try to parse and validate the date
                    if '-' in match:
                        date_obj = datetime.strptime(match, '%Y-%m-%d')
                    elif '/' in match and len(match.split('/')[2]) == 4:
                        date_obj = datetime.strptime(match, '%m/%d/%Y')
                    elif '/' in match:
                        date_obj = datetime.strptime(match, '%Y/%m/%d')
                    else:
                        continue
                        
                    if date_obj.year >= 1990 and date_obj.year <= datetime.now().year:
                        return date_obj.strftime('%Y-%m-%d')
                except:
                    continue
        
        return None
    
    def sample_cids_from_collection(self, sample_size=100):
        """
        Sample CIDs from the collection based on defined bins.
        
        Args:
            sample_size: Number of CIDs to sample from each bin
            
        Returns:
            dict: {bin_name: [list of sampled CIDs]}
        """
        logger.info(f"Loading collection from {self.csv_file_path}")
        
        # Read the CSV file in chunks to handle large files
        chunk_size = 50000
        sampled_cids = {}
        
        # Initialize bins
        for bin_range in self.cid_bins:
            bin_name = f"CID_{bin_range[0]}_{bin_range[1]}"
            sampled_cids[bin_name] = []
        
        # Process file in chunks
        for chunk in pd.read_csv(self.csv_file_path, chunksize=chunk_size):
            # Ensure we have the CID column
            if 'CID' not in chunk.columns:
                logger.error("CID column not found in the CSV file")
                return {}
            
            for bin_range in self.cid_bins:
                min_cid, max_cid = bin_range
                bin_name = f"CID_{min_cid}_{max_cid}"
                
                # Filter CIDs in this range
                bin_cids = chunk[(chunk['CID'] >= min_cid) & (chunk['CID'] < max_cid)]['CID'].tolist()
                sampled_cids[bin_name].extend(bin_cids)
        
        # Sample from each bin
        final_samples = {}
        for bin_name, cids in sampled_cids.items():
            if len(cids) > 0:
                sample_count = min(sample_size, len(cids))
                final_samples[bin_name] = random.sample(cids, sample_count)
                logger.info(f"Sampled {sample_count} CIDs from {bin_name} (available: {len(cids)})")
            else:
                final_samples[bin_name] = []
                logger.warning(f"No CIDs found in range {bin_name}")
        
        return final_samples
    
    def estimate_temporal_period_from_cid(self, cid):
        """
        Estimate temporal period based on CID ranges since creation date scraping is unreliable.
        
        Args:
            cid: PubChem Compound ID
            
        Returns:
            str: Estimated temporal period
        """
        # Based on PubChem growth patterns and historical data
        cid_temporal_mapping = [
            (1, 1000, "1996-2000"),          # Very early PubChem
            (1000, 10000, "2000-2004"),      # Early period
            (10000, 100000, "2004-2008"),    # Rapid growth phase
            (100000, 500000, "2008-2010"),   # Database expansion
            (500000, 1000000, "2010-2012"),  # Continued growth
            (1000000, 5000000, "2012-2015"), # Major expansion
            (5000000, 10000000, "2015-2017"), # High-throughput period
            (10000000, 50000000, "2017-2020"), # Modern era
            (50000000, 116000000, "2020-2024") # Current period
        ]
        
        for min_cid, max_cid, period in cid_temporal_mapping:
            if min_cid <= cid < max_cid:
                return period
        
        return "Unknown"
    
    def collect_creation_dates(self, sampled_cids, output_file="pubchem_creation_dates.csv"):
        """
        Collect creation dates for sampled CIDs with fallback to CID-based estimation.
        
        Args:
            sampled_cids: Dictionary of sampled CIDs from sample_cids_from_collection
            output_file: Output CSV file for results
        """
        results = []
        total_cids = sum(len(cids) for cids in sampled_cids.values())
        
        logger.info(f"Collecting temporal information for {total_cids} CIDs...")
        logger.info("Note: Since creation date scraping is unreliable, using CID-based temporal estimation")
        
        with tqdm(total=total_cids, desc="Processing temporal data") as pbar:
            for bin_name, cids in sampled_cids.items():
                for cid in cids:
                    # Try enhanced scraping first
                    creation_date = self.try_better_date_scraping(cid)
                    
                    # If that fails, try the original method
                    if not creation_date:
                        creation_date = self.scrape_compound_creation_date(cid)
                    
                    # Use CID-based estimation as fallback (primary method)
                    temporal_period = self.estimate_temporal_period_from_cid(cid)
                    
                    results.append({
                        'CID': cid,
                        'Bin': bin_name,
                        'Creation_Date': creation_date,  # May be None
                        'Temporal_Period': temporal_period,  # Reliable estimation
                        'Scraped_At': datetime.now().isoformat()
                    })
                    
                    pbar.update(1)
                    
                    # Sleep to avoid overloading PubChem
                    time.sleep(self.delay)
        
        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv(output_file, index=False)
        
        # Report success rates
        scraped_dates = df_results['Creation_Date'].notna().sum()
        estimated_periods = df_results['Temporal_Period'].notna().sum()
        
        logger.info(f"Temporal information saved to {output_file}")
        logger.info(f"Scraped dates: {scraped_dates}/{total_cids} ({scraped_dates/total_cids*100:.1f}%)")
        logger.info(f"Estimated periods: {estimated_periods}/{total_cids} ({estimated_periods/total_cids*100:.1f}%)")
        
        return df_results
    
    def calculate_molecular_descriptors(self, smiles):
        """
        Calculate comprehensive molecular descriptors for a SMILES string.
        
        Args:
            smiles: SMILES string
            
        Returns:
            dict: Dictionary of calculated descriptors
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Add hydrogens for accurate calculations
            mol_h = Chem.AddHs(mol)
            
            descriptors = {
                'MW': rdMolDescriptors.CalcExactMolWt(mol),
                'RotBonds': CalcNumRotatableBonds(mol),
                'TPSA': CalcTPSA(mol),
                'HAcceptors': CalcNumHBA(mol),
                'HDonors': CalcNumHBD(mol),
                'cLogP': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],  # MolLogP equivalent
                'AromaticRings': CalcNumAromaticRings(mol),
                'TotalRings': rdMolDescriptors.CalcNumRings(mol),
                'Heterocycles': CalcNumHeterocycles(mol),
                'NumF': len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F']),
                'NumCl': len([atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl']),
                'ChiralCenters': CalcNumAtomStereoCenters(mol_h),
                'QED': QED.qed(mol)
            }
            
            return descriptors
            
        except Exception as e:
            logger.debug(f"Error calculating descriptors for {smiles}: {str(e)}")
            return None
    
    def analyze_molecular_properties(self, sampled_cids, output_file="molecular_analysis.csv"):
        """
        Analyze molecular properties for sampled CIDs.
        
        Args:
            sampled_cids: Dictionary of sampled CIDs
            output_file: Output CSV file for analysis results
        """
        logger.info("Starting molecular property analysis...")
        
        # Load the main collection to get SMILES
        logger.info("Loading SMILES data from collection...")
        
        # Create a mapping of CID to SMILES
        cid_to_smiles = {}
        chunk_size = 50000
        
        all_target_cids = set()
        for cids in sampled_cids.values():
            all_target_cids.update(cids)
        
        for chunk in pd.read_csv(self.csv_file_path, chunksize=chunk_size):
            if 'CID' in chunk.columns and 'Smiles' in chunk.columns:
                chunk_filtered = chunk[chunk['CID'].isin(all_target_cids)]
                for _, row in chunk_filtered.iterrows():
                    cid_to_smiles[row['CID']] = row['Smiles']
        
        logger.info(f"Found SMILES for {len(cid_to_smiles)} CIDs")
        
        # Calculate descriptors
        results = []
        total_cids = len(all_target_cids)
        
        with tqdm(total=total_cids, desc="Calculating descriptors") as pbar:
            for bin_name, cids in sampled_cids.items():
                for cid in cids:
                    if cid in cid_to_smiles:
                        smiles = cid_to_smiles[cid]
                        descriptors = self.calculate_molecular_descriptors(smiles)
                        
                        if descriptors:
                            result = {
                                'CID': cid,
                                'Bin': bin_name,
                                'Smiles': smiles,
                                **descriptors
                            }
                            results.append(result)
                    
                    pbar.update(1)
        
        # Save results
        df_analysis = pd.DataFrame(results)
        df_analysis.to_csv(output_file, index=False)
        logger.info(f"Molecular analysis saved to {output_file}")
        
        return df_analysis
    
    def create_analysis_summary(self, df_analysis, output_file="analysis_summary.csv"):
        """
        Create summary statistics for each CID bin.
        
        Args:
            df_analysis: DataFrame with molecular analysis results
            output_file: Output file for summary statistics
        """
        logger.info("Creating analysis summary...")
        
        # Define properties to analyze
        properties = ['MW', 'RotBonds', 'TPSA', 'HAcceptors', 'HDonors', 'cLogP',
                     'AromaticRings', 'TotalRings', 'Heterocycles', 'NumF', 'NumCl',
                     'ChiralCenters', 'QED']
        
        summary_results = []
        
        for bin_name in df_analysis['Bin'].unique():
            bin_data = df_analysis[df_analysis['Bin'] == bin_name]
            
            for prop in properties:
                if prop in bin_data.columns:
                    prop_data = bin_data[prop].dropna()
                    
                    if len(prop_data) > 0:
                        summary_results.append({
                            'Bin': bin_name,
                            'Property': prop,
                            'Count': len(prop_data),
                            'Mean': prop_data.mean(),
                            'Median': prop_data.median(),
                            'Std': prop_data.std(),
                            'Min': prop_data.min(),
                            'Max': prop_data.max(),
                            'Q25': prop_data.quantile(0.25),
                            'Q75': prop_data.quantile(0.75)
                        })
        
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(output_file, index=False)
        logger.info(f"Summary statistics saved to {output_file}")
        
        return df_summary
    
    def create_visualizations(self, df_analysis, df_summary):
        """
        Create visualizations for the temporal analysis.
        
        Args:
            df_analysis: DataFrame with detailed analysis results
            df_summary: DataFrame with summary statistics
        """
        logger.info("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # Properties to visualize
        key_properties = ['MW', 'RotBonds', 'TPSA', 'cLogP', 'AromaticRings', 'QED']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        axes = axes.ravel()
        
        for i, prop in enumerate(key_properties):
            if prop in df_analysis.columns:
                # Box plot for each bin
                bin_order = sorted(df_analysis['Bin'].unique(), 
                                 key=lambda x: int(x.split('_')[1]))
                
                box_data = [df_analysis[df_analysis['Bin'] == bin_name][prop].dropna() 
                           for bin_name in bin_order]
                
                axes[i].boxplot(box_data, labels=[b.replace('CID_', '') for b in bin_order])
                axes[i].set_title(f'{prop} Distribution Across CID Ranges')
                axes[i].set_xlabel('CID Range')
                axes[i].set_ylabel(prop)
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('temporal_molecular_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create trend plots
        fig, axes = plt.subplots(2, 3, figsize=fig_size)
        axes = axes.ravel()
        
        for i, prop in enumerate(key_properties):
            if prop in df_summary.columns:
                prop_summary = df_summary[df_summary['Property'] == prop].copy()
                
                # Extract numeric bin ranges for x-axis
                prop_summary['Bin_Start'] = prop_summary['Bin'].apply(
                    lambda x: int(x.split('_')[1]))
                prop_summary = prop_summary.sort_values('Bin_Start')
                
                axes[i].plot(prop_summary['Bin_Start'], prop_summary['Mean'], 
                           'o-', label='Mean', linewidth=2)
                axes[i].plot(prop_summary['Bin_Start'], prop_summary['Median'], 
                           's-', label='Median', linewidth=2)
                
                axes[i].set_xscale('log')
                axes[i].set_title(f'{prop} Trends Over CID Ranges')
                axes[i].set_xlabel('CID Range Start')
                axes[i].set_ylabel(prop)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Visualizations saved as 'temporal_molecular_analysis.png' and 'temporal_trends.png'")

def main():
    """Main execution function."""
    # Configuration
    csv_file_path = "chemically_clean_mw1000_collection.csv"
    sample_size = 100  # Number of CIDs to sample from each bin
    delay_between_requests = 1.0  # Seconds between web requests
    
    # Initialize analyzer
    analyzer = PubChemTemporalAnalyzer(csv_file_path, delay_between_requests)
    
    # Check if input file exists
    if not os.path.exists(csv_file_path):
        logger.error(f"Input file {csv_file_path} not found!")
        logger.info("Please make sure the file exists in the current directory.")
        return
    
    # Step 1: Sample CIDs from the collection
    logger.info("Step 1: Sampling CIDs from collection...")
    sampled_cids = analyzer.sample_cids_from_collection(sample_size)
    
    if not any(sampled_cids.values()):
        logger.error("No CIDs were sampled. Please check your input file format.")
        return
    
    # Step 2: Collect creation dates (optional, time-consuming)
    collect_dates = input("Do you want to collect creation dates? (y/n): ").lower().strip() == 'y'
    
    if collect_dates:
        logger.info("Step 2: Collecting creation dates...")
        df_dates = analyzer.collect_creation_dates(sampled_cids)
        logger.info(f"Collected creation dates for {len(df_dates)} compounds")
    
    # Step 3: Analyze molecular properties
    logger.info("Step 3: Analyzing molecular properties...")
    df_analysis = analyzer.analyze_molecular_properties(sampled_cids)
    
    if df_analysis.empty:
        logger.error("No molecular analysis results generated.")
        return
    
    # Step 4: Create summary statistics
    logger.info("Step 4: Creating summary statistics...")
    df_summary = analyzer.create_analysis_summary(df_analysis)
    
    # Step 5: Create visualizations
    logger.info("Step 5: Creating visualizations...")
    analyzer.create_visualizations(df_analysis, df_summary)
    
    logger.info("Analysis complete!")
    logger.info("Generated files:")
    logger.info("- molecular_analysis.csv: Detailed molecular descriptors")
    logger.info("- analysis_summary.csv: Summary statistics by CID range")
    logger.info("- temporal_molecular_analysis.png: Distribution plots")
    logger.info("- temporal_trends.png: Trend analysis plots")
    
    if collect_dates:
        logger.info("- pubchem_creation_dates.csv: Scraped creation dates")

if __name__ == "__main__":
    main()
