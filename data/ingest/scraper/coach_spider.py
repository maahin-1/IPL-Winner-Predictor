"""
T5: Scrapy spider for coach win records and venue condition reports.
Targets: HowStat, CricketArchive, franchise press releases.
Respects robots.txt per PRD data_sources[T5].notes.
"""
import scrapy
from scrapy import Spider
from scrapy.http import Response


class CoachStatsSpider(Spider):
    name = "coach_stats"
    allowed_domains = ["howstat.com", "cricketarchive.com"]
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 2,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 1.0,
    }

    def start_requests(self):
        coaches_url = "https://www.howstat.com/cricket/Statistics/IPL/CoachList.asp"
        yield scrapy.Request(coaches_url, callback=self.parse_coach_list)

    def parse_coach_list(self, response: Response):
        for row in response.css("table.tabledata tr:not(:first-child)"):
            cells = row.css("td::text").getall()
            if len(cells) >= 4:
                yield {
                    "coach_name": cells[0].strip(),
                    "team": cells[1].strip(),
                    "seasons": cells[2].strip(),
                    "win_rate": cells[3].strip(),
                    "source": "howstat",
                }


class VenueConditionsSpider(Spider):
    name = "venue_conditions"
    allowed_domains = ["cricketarchive.com"]
    custom_settings = {
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 3,
    }

    def start_requests(self):
        venues_url = "https://cricketarchive.com/Archive/Grounds/index_I.html"
        yield scrapy.Request(venues_url, callback=self.parse_venue_list)

    def parse_venue_list(self, response: Response):
        for link in response.css("a[href*='Ground']"):
            venue_name = link.css("::text").get("").strip()
            href = link.attrib.get("href", "")
            if venue_name and href:
                yield response.follow(href, callback=self.parse_venue_detail,
                                      meta={"venue": venue_name})

    def parse_venue_detail(self, response: Response):
        venue = response.meta["venue"]
        stats = response.css("table.tabledata tr")
        for row in stats:
            cells = row.css("td::text").getall()
            if len(cells) >= 3:
                yield {
                    "venue": venue,
                    "stat_name": cells[0].strip(),
                    "value": cells[1].strip(),
                    "source": "cricketarchive",
                }
