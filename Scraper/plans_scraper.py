import os
import json
import time
import random
import csv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from rich.console import Console
from rich.table import Table
from rich import box
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create RAG directory if it doesn't exist
os.makedirs("RAG", exist_ok=True)

console = Console()

class AirtelScraper:
    def __init__(self):
        self.driver = None
        self.all_plans = []
        self.seen_plans = set()
        self.total_plans = {"Prepaid": 0, "Postpaid": 0, "Broadband": 0, "DTH": 0, "Black": 0}
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0"
        ]
        
    def setup_driver(self):
        """Setup Chrome driver with advanced anti-detection options"""
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-web-security")
        options.add_argument("--disable-features=VizDisplayCompositor")
        options.add_argument("--disable-notifications")
        options.add_argument("--disable-popup-blocking")
        options.add_argument(f"--user-agent={random.choice(self.user_agents)}")
        
        # Experimental options to avoid detection
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--disable-blink-features=AutomationControlled')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            
            # Mask selenium parameters
            self.driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": random.choice(self.user_agents)
            })
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            self.driver.execute_script("window.navigator.chrome = {runtime: {}, etc: 'etc'};")
            self.driver.execute_script(
                "const originalQuery = window.navigator.permissions.query;"
                "window.navigator.permissions.query = (parameters) => ("
                "    parameters.name === 'notifications' ? "
                "        Promise.resolve({ state: Notification.permission }) :"
                "        originalQuery(parameters)"
                ");"
            )
            
            return True
        except Exception as e:
            console.print(f"[red]Error setting up driver: {e}")
            return False

    def human_like_delay(self, min=0.5, max=1.5):
        """Random delay to mimic human behavior"""
        time.sleep(random.uniform(min, max))
    
    def safe_find_element(self, by, value, timeout=15, element=None):
        """Safely find element with timeout and retry"""
        attempt = 0
        while attempt < 3:
            try:
                if element:
                    return WebDriverWait(element, timeout).until(
                        EC.presence_of_element_located((by, value)))
                else:
                    return WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((by, value)))
            except (TimeoutException, StaleElementReferenceException):
                attempt += 1
                self.human_like_delay(0.2, 0.5)
        return None
    
    def safe_find_elements(self, by, value, timeout=15, element=None):
        """Safely find elements with timeout and retry"""
        attempt = 0
        while attempt < 3:
            try:
                if element:
                    WebDriverWait(element, timeout).until(
                        EC.presence_of_element_located((by, value)))
                    return element.find_elements(by, value)
                else:
                    WebDriverWait(self.driver, timeout).until(
                        EC.presence_of_element_located((by, value)))
                    return self.driver.find_elements(by, value)
            except (TimeoutException, StaleElementReferenceException):
                attempt += 1
                self.human_like_delay(0.2, 0.5)
        return []
    
    def extract_text_safe(self, element, xpath_list):
        """Safely extract text from element using multiple xpath options"""
        if not element:
            return "N/A"
        
        for xpath in xpath_list:
            try:
                found_elements = element.find_elements(By.XPATH, xpath)
                if found_elements:
                    text = found_elements[0].text.strip()
                    if text and text != "N/A":
                        return text
            except:
                continue
        return "N/A"
    
    def extract_price(self, element):
        """Robust price extraction for Airtel's dynamic pricing"""
        # Strategy 1: Look for rupee symbol
        for xpath in [
            ".//*[contains(text(), '₹')]",
            ".//*[contains(text(), 'Rs')]",
            ".//*[contains(text(), 'INR')]"
        ]:
            try:
                elements = element.find_elements(By.XPATH, xpath)
                for elem in elements:
                    text = elem.text.strip()
                    if '₹' in text:
                        return text
            except:
                continue
        
        # Strategy 2: Check common price classes
        for cls in ["price", "amount", "cost", "rupee", "currency"]:
            try:
                price_elem = element.find_element(By.XPATH, f".//*[contains(@class, '{cls}')]")
                text = price_elem.text.strip()
                if text:
                    return text
            except:
                continue
        
        # Strategy 3: Look for numeric values with common plan structures
        try:
            heading = element.find_element(By.XPATH, ".//h3 | .//h4 | .//h5 | .//div[contains(@class, 'heading')]")
            text = heading.text.strip()
            if '₹' in text:
                return text
        except:
            pass
        
        return "N/A"
    
    def close_popups(self):
        """Close any popups that might interfere with scraping"""
        try:
            # Location popup
            location_btn = self.safe_find_element(
                By.XPATH, "//button[contains(.,'Enter your location')]", 3)
            if location_btn:
                location_btn.click()
                self.human_like_delay()
                
            # Cookie banner
            cookie_accept = self.safe_find_element(
                By.XPATH, "//button[contains(.,'Accept') or contains(.,'Allow')]", 3)
            if cookie_accept:
                cookie_accept.click()
                self.human_like_delay()
                
            # Notification popups
            notif_dismiss = self.safe_find_element(
                By.XPATH, "//button[contains(.,'Later') or contains(.,'Not Now')]", 3)
            if notif_dismiss:
                notif_dismiss.click()
                self.human_like_delay()
                
            # Overlay close buttons
            overlay_close = self.safe_find_element(
                By.XPATH, "//button[contains(@class, 'close') or contains(@class, 'dismiss')]", 3)
            if overlay_close:
                overlay_close.click()
                self.human_like_delay()
                
        except Exception as e:
            logger.debug(f"Popup handling issue: {str(e)}")
    
    def scroll_to_element(self, element):
        """Scroll to element with human-like behavior"""
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", element)
            self.human_like_delay(0.3, 0.8)
            # Additional micro-movement to trigger lazy loading
            self.driver.execute_script("window.scrollBy(0, 50);")
            self.human_like_delay(0.1, 0.3)
            self.driver.execute_script("window.scrollBy(0, -20);")
        except:
            pass
    
    def scrape_prepaid_plans(self):
        """Scrape Mobile Prepaid plans with enhanced interaction"""
        console.print("[bold blue]Scraping Mobile Prepaid Plans...[/]")
        
        try:
            self.driver.get("https://www.airtel.in/recharge-online")
            time.sleep(3)
            self.close_popups()
            
            # Wait for page to load
            if not self.safe_find_element(By.TAG_NAME, "body", 20):
                console.print("[red]Failed to load prepaid page")
                return
            
            # Get all category buttons
            category_buttons = self.safe_find_elements(
                By.XPATH, "//div[contains(@class, 'tabs')]//button[contains(@class, 'tab')]", 10
            )
            
            if not category_buttons:
                console.print("[yellow]No category buttons found, scraping all visible plans")
                self.scrape_plan_cards("Mobile Prepaid", "All Plans")
                return
            
            console.print(f"[green]Found {len(category_buttons)} categories")
            
            # Process each category button
            for button in category_buttons:
                try:
                    category_name = button.text.strip()
                    if not category_name or category_name in ["Popular", "Recommended", "All"]:
                        continue
                    
                    console.print(f"[cyan]Processing category: {category_name}")
                    
                    # Scroll to button
                    self.scroll_to_element(button)
                    
                    # Click using JavaScript to avoid interception
                    self.driver.execute_script("arguments[0].click();", button)
                    self.human_like_delay(1, 2)  # Wait for content to load
                    
                    # Scrape plans for this specific category
                    self.scrape_plan_cards("Mobile Prepaid", category_name)
                    
                except Exception as e:
                    console.print(f"[red]Error processing category {category_name}: {e}")
                    continue
        
        except Exception as e:
            console.print(f"[red]Error in prepaid scraping: {e}")
    
    def scrape_postpaid_plans(self):
        """Scrape Mobile Postpaid plans with improved navigation"""
        console.print("[bold blue]Scraping Mobile Postpaid Plans...[/]")
        
        try:
            self.driver.get("https://www.airtel.in/myplan-infinity/postpaid")
            time.sleep(3)
            self.close_popups()
            
            # Wait for page to load
            if not self.safe_find_element(By.TAG_NAME, "body", 20):
                console.print("[red]Failed to load postpaid page")
                return
            
            # Handle plan type toggles
            toggles = self.safe_find_elements(
                By.XPATH, "//div[contains(@class, 'toggle')]//button", 5)
            
            if toggles:
                for toggle in toggles:
                    toggle_name = toggle.text.strip()
                    if not toggle_name:
                        continue
                    
                    console.print(f"[cyan]Processing toggle: {toggle_name}")
                    
                    self.scroll_to_element(toggle)
                    self.driver.execute_script("arguments[0].click();", toggle)
                    self.human_like_delay(1, 2)  # Wait for content to update
                    
                    # Scrape plans for this toggle
                    self.scrape_plan_cards("Mobile Postpaid", toggle_name)
            else:
                self.scrape_plan_cards("Mobile Postpaid", "Postpaid Plans")
            
        except Exception as e:
            console.print(f"[red]Error in postpaid scraping: {e}")
    
    def scrape_broadband_plans(self):
        """Scrape Broadband plans with location handling"""
        console.print("[bold blue]Scraping Broadband Plans...[/]")
        
        try:
            self.driver.get("https://www.airtel.in/broadband")
            time.sleep(3)
            self.close_popups()
            
            # Handle location input
            location_input = self.safe_find_element(
                By.XPATH, "//input[@placeholder='Enter your location']", 5)
            if location_input:
                location_input.send_keys("Delhi")
                self.human_like_delay(0.5, 1)
                location_input.send_keys(Keys.RETURN)
                self.human_like_delay(1, 2)
            
            if not self.safe_find_element(By.TAG_NAME, "body", 20):
                console.print("[red]Failed to load broadband page")
                return
            
            self.scrape_plan_cards("Broadband", "Fiber Plans")
            
        except Exception as e:
            console.print(f"[red]Error in broadband scraping: {e}")
    
    def scrape_dth_plans(self):
        """Scrape DTH plans with robust pagination"""
        console.print("[bold blue]Scraping DTH Plans...[/]")
        
        try:
            self.driver.get("https://www.airtel.in/dth")
            time.sleep(3)
            self.close_popups()
            
            if not self.safe_find_element(By.TAG_NAME, "body", 20):
                console.print("[red]Failed to load DTH page")
                return
            
            # Handle pagination
            page_count = 1
            while True:
                console.print(f"[cyan]Processing page {page_count}")
                self.scrape_plan_cards("DTH", f"DTH Plans - Page {page_count}")
                
                # Find next page button
                next_btn = self.safe_find_element(
                    By.XPATH, "//button[contains(.,'Next') or contains(@class,'pagination-next')]", 5)
                
                if not next_btn or "disabled" in next_btn.get_attribute("class"):
                    break
                
                self.scroll_to_element(next_btn)
                self.driver.execute_script("arguments[0].click();", next_btn)
                self.human_like_delay(2, 3)  # Wait for page load
                page_count += 1
                
                # Safety break
                if page_count > 15:
                    break
            
        except Exception as e:
            console.print(f"[red]Error in DTH scraping: {e}")
    
    def scrape_black_plans(self):
        """Scrape Airtel Black plans"""
        console.print("[bold blue]Scraping Airtel Black Plans...[/]")
        
        try:
            self.driver.get("https://www.airtel.in/airtel-black")
            time.sleep(3)
            self.close_popups()
            
            if not self.safe_find_element(By.TAG_NAME, "body", 20):
                console.print("[red]Failed to load Airtel Black page")
                return
            
            self.scrape_plan_cards("Airtel Black", "Black Plans")
            
        except Exception as e:
            console.print(f"[red]Error in Airtel Black scraping: {e}")
    
    def scrape_plan_cards(self, plan_type, category):
        """Robust plan card scraping with lazy loading handling"""
        console.print(f"[dim]Scraping {category} under {plan_type}...")
        
        # Scroll to load lazy content
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        self.human_like_delay(0.5, 1)
        self.driver.execute_script("window.scrollTo(0, 0);")
        self.human_like_delay(0.5, 1)
        
        # Updated selectors with fallbacks
        card_selectors = [
            "//div[contains(@class, 'planCard')]",
            "//div[contains(@class, 'packCard')]",
            "//div[contains(@class, 'card-container')]",
            "//div[contains(@class, 'offer-card')]",
            "//div[contains(@class, 'product-card')]"
        ]
        
        plan_cards = []
        for selector in card_selectors:
            plan_cards = self.safe_find_elements(By.XPATH, selector, 15)
            if plan_cards:
                console.print(f"[dim]Using selector: {selector}")
                break
        
        if not plan_cards:
            console.print(f"[yellow]No plan cards found for {plan_type} - {category}")
            return
        
        console.print(f"[green]Found {len(plan_cards)} plan cards for {category}")
        
        category_count = 0
        
        for i, card in enumerate(plan_cards):
            try:
                # Ensure card is in view
                self.scroll_to_element(card)
                
                # Extract details
                price = self.extract_price(card)
                
                # Data allowance
                data = self.extract_text_safe(card, [
                    ".//*[contains(., 'GB') or contains(., 'MB') or contains(., 'Data')]",
                    ".//div[contains(@class, 'data')]",
                    ".//div[contains(@class, 'internet')]"
                ])
                
                # Validity
                validity = self.extract_text_safe(card, [
                    ".//*[contains(., 'days') or contains(., 'validity') or contains(., 'month')]",
                    ".//div[contains(@class, 'validity')]",
                    ".//div[contains(@class, 'duration')]"
                ])
                
                # Calls
                calls = self.extract_text_safe(card, [
                    ".//*[contains(., 'calls') or contains(., 'calling') or contains(., 'voice')]",
                    ".//div[contains(@class, 'calls')]",
                    ".//div[contains(@class, 'voice')]"
                ])
                
                # SMS
                sms = self.extract_text_safe(card, [
                    ".//*[contains(., 'SMS')]",
                    ".//div[contains(@class, 'sms')]"
                ])
                
                # Benefits
                benefits = []
                try:
                    # Multiple benefit extraction strategies
                    benefit_containers = card.find_elements(
                        By.XPATH, ".//ul[contains(@class, 'benefits')] | .//div[contains(@class, 'features')] | .//div[contains(@class, 'benefits')]"
                    )
                    
                    for container in benefit_containers:
                        items = container.find_elements(By.XPATH, ".//li | .//div")
                        for item in items:
                            text = item.text.strip()
                            if text and len(text) > 3 and text not in benefits:
                                benefits.append(text)
                except:
                    pass
                
                benefits_str = " | ".join(benefits[:5]) if benefits else "N/A"
                
                # Create unique plan key
                plan_key = f"{plan_type}_{category}_{price}_{data}_{validity}".lower()
                if plan_key in self.seen_plans:
                    continue
                self.seen_plans.add(plan_key)
                
                # Add to results
                plan_data = {
                    "plan_type": plan_type,
                    "category": category,
                    "price": price,
                    "data": data,
                    "validity": validity,
                    "calls": calls,
                    "sms": sms,
                    "benefits": benefits_str
                }
                
                self.all_plans.append(plan_data)
                
                # Update counters
                self.total_plans[plan_type.split()[-1]] = self.total_plans.get(plan_type.split()[-1], 0) + 1
                category_count += 1
                
                console.print(f"[dim]Plan {i+1}: {price} | {data} | {validity}")
                
            except Exception as e:
                logger.error(f"Error processing plan card {i}: {str(e)}")
                continue
        
        console.print(f"[green]Scraped {category_count} plans from {category}")
    
    def save_data(self):
        """Save scraped data to CSV and JSON files"""
        if not self.all_plans:
            console.print("[red]No plans to save!")
            return
        
        # Save to CSV
        csv_filename = "RAG/airtel_all_plans.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['plan_type', 'category', 'price', 'data', 'validity', 'calls', 'sms', 'benefits']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.all_plans)
        
        # Save to JSON
        json_filename = "RAG/airtel_all_plans.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(self.all_plans, jsonfile, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ Data saved to {csv_filename} and {json_filename}")
    
    def run(self):
        """Main scraping function with performance monitoring"""
        console.rule("[bold blue]Airtel Plans Scraper[/]", style="blue")
        start_time = time.time()
        
        if not self.setup_driver():
            return
        
        try:
            # Scrape all plan types
            self.scrape_prepaid_plans()
            self.scrape_postpaid_plans()
            self.scrape_broadband_plans()
            self.scrape_dth_plans()
            self.scrape_black_plans()
            
            # Save data
            self.save_data()
            
            # Display summary
            console.rule("[bold green]Scraping Complete[/]", style="green")
            
            summary_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
            summary_table.add_column("Plan Type", style="cyan")
            summary_table.add_column("Plans Found", justify="right")
            
            for plan_type, count in self.total_plans.items():
                summary_table.add_row(plan_type, str(count))
            
            console.print(summary_table)
            console.print(f"\n[bold green]Total plans scraped: {len(self.all_plans)}")
            console.print(f"[dim]Execution time: {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            console.print(f"[red]Fatal error during scraping: {e}")
            logger.exception("Scraping failed")
        
        finally:
            if self.driver:
                self.driver.quit()
                console.print("[yellow]Browser closed.")

if __name__ == "__main__":
    scraper = AirtelScraper()
    scraper.run()