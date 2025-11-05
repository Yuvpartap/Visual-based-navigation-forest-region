import os
import torch
import logging
import math 
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleMapsTrajectoryPlotter:
    """Converts trajectory from tile coordinates to lat/lon and creates interactive map."""
    
    @staticmethod
    def tile_to_latlon(tile_x, tile_y, zoom):
        """
        Convert tile coordinates to latitude/longitude.
        Uses Web Mercator projection (EPSG:3857).
        
        Args:
            tile_x: Tile X coordinate
            tile_y: Tile Y coordinate
            zoom: Zoom level
            
        Returns:
            Tuple of (latitude, longitude)
        """
        n = 2.0 ** zoom
        lon_deg = tile_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * tile_y / n)))
        lat_deg = math.degrees(lat_rad)
        return lat_deg, lon_deg
    
    @staticmethod
    def create_google_maps_html(centers, zoom, save_dir, api_key=None):
        """
        Create an interactive Google Maps HTML file with trajectory.
        
        Args:
            centers: List of (tile_x, tile_y) coordinates
            zoom: Zoom level used for tiles
            save_dir: Directory to save HTML file
            api_key: Optional Google Maps API key (if None, uses basic map)
        """
        if not centers or len(centers) < 2:
            logger.warning("✗ Insufficient trajectory points for Google Maps visualization")
            return
        
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING GOOGLE MAPS TRAJECTORY")
        logger.info("=" * 70)
        
        # Convert all tile coordinates to lat/lon
        latlon_points = []
        for tile_x, tile_y in centers:
            lat, lon = GoogleMapsTrajectoryPlotter.tile_to_latlon(tile_x, tile_y, zoom)
            latlon_points.append({'lat': lat, 'lng': lon})
        
        # Calculate center point for map
        center_lat = sum(p['lat'] for p in latlon_points) / len(latlon_points)
        center_lng = sum(p['lng'] for p in latlon_points) / len(latlon_points)
        
        # Create HTML with embedded JavaScript
        html_content = GoogleMapsTrajectoryPlotter._generate_html(
            latlon_points, center_lat, center_lng, api_key
        )
        
        # Save HTML file
        html_path = os.path.join(save_dir, "trajectory_google_maps.html")
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"✓ Google Maps trajectory saved: {html_path}")
        logger.info(f"  Total points: {len(latlon_points)}")
        logger.info(f"  Center: {center_lat:.6f}, {center_lng:.6f}")
        logger.info(f"  Start: {latlon_points[0]['lat']:.6f}, {latlon_points[0]['lng']:.6f}")
        logger.info(f"  End: {latlon_points[-1]['lat']:.6f}, {latlon_points[-1]['lng']:.6f}")
        logger.info("=" * 70)
        
        # Also save lat/lon data as JSON for other uses
        json_path = os.path.join(save_dir, "trajectory_latlon.json")
        with open(json_path, 'w') as f:
            json.dump({
                'points': latlon_points,
                'center': {'lat': center_lat, 'lng': center_lng},
                'zoom_level': zoom
            }, f, indent=2)
        logger.info(f"✓ Lat/Lon data saved: {json_path}")
    
    @staticmethod
    def _generate_html(points, center_lat, center_lng, api_key):
        """Generate HTML content with embedded Google Maps."""
        
        # If no API key, use Leaflet with OpenStreetMap (free alternative)
        if api_key is None:
            return GoogleMapsTrajectoryPlotter._generate_leaflet_html(
                points, center_lat, center_lng
            )
        
        # Google Maps version (requires API key)
        points_json = json.dumps(points)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Drone Trajectory - Google Maps</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }}
        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #1a73e8;
        }}
        .info-panel p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .legend {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <h3>🛩️ Drone Trajectory</h3>
        <p><strong>Total Points:</strong> <span id="point-count">{len(points)}</span></p>
        <p><strong>Start:</strong> <span id="start-coords"></span></p>
        <p><strong>End:</strong> <span id="end-coords"></span></p>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #4285F4; height: 3px;"></div>
                <span>Flight Path</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #34A853; width: 12px; height: 12px; border-radius: 50%;"></div>
                <span>Start Point</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #EA4335; width: 12px; height: 12px; border-radius: 50%;"></div>
                <span>End Point</span>
            </div>
        </div>
    </div>
    
    <div id="map"></div>
    
    <script>
        let map;
        let flightPath;
        const trajectoryPoints = {points_json};
        
        function initMap() {{
            // Initialize map centered on trajectory
            map = new google.maps.Map(document.getElementById('map'), {{
                center: {{ lat: {center_lat}, lng: {center_lng} }},
                zoom: 16,
                mapTypeId: 'satellite', // Use satellite view
                mapTypeControl: true,
                mapTypeControlOptions: {{
                    style: google.maps.MapTypeControlStyle.HORIZONTAL_BAR,
                    position: google.maps.ControlPosition.TOP_RIGHT,
                    mapTypeIds: ['roadmap', 'satellite', 'hybrid', 'terrain']
                }}
            }});
            
            // Draw the flight path
            flightPath = new google.maps.Polyline({{
                path: trajectoryPoints,
                geodesic: true,
                strokeColor: '#4285F4',
                strokeOpacity: 0.8,
                strokeWeight: 3
            }});
            flightPath.setMap(map);
            
            // Add start marker (green)
            new google.maps.Marker({{
                position: trajectoryPoints[0],
                map: map,
                title: 'Start',
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: '#34A853',
                    fillOpacity: 1,
                    strokeColor: 'white',
                    strokeWeight: 2
                }}
            }});
            
            // Add end marker (red)
            new google.maps.Marker({{
                position: trajectoryPoints[trajectoryPoints.length - 1],
                map: map,
                title: 'End',
                icon: {{
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 8,
                    fillColor: '#EA4335',
                    fillOpacity: 1,
                    strokeColor: 'white',
                    strokeWeight: 2
                }}
            }});
            
            // Add waypoint markers (every 10th point)
            for (let i = 0; i < trajectoryPoints.length; i += 10) {{
                if (i === 0 || i === trajectoryPoints.length - 1) continue;
                
                new google.maps.Marker({{
                    position: trajectoryPoints[i],
                    map: map,
                    title: `Waypoint ${{i}}`,
                    icon: {{
                        path: google.maps.SymbolPath.CIRCLE,
                        scale: 4,
                        fillColor: '#FBBC04',
                        fillOpacity: 0.6,
                        strokeColor: 'white',
                        strokeWeight: 1
                    }}
                }});
            }}
            
            // Update info panel
            document.getElementById('start-coords').textContent = 
                `${{trajectoryPoints[0].lat.toFixed(6)}}, ${{trajectoryPoints[0].lng.toFixed(6)}}`;
            document.getElementById('end-coords').textContent = 
                `${{trajectoryPoints[trajectoryPoints.length-1].lat.toFixed(6)}}, ${{trajectoryPoints[trajectoryPoints.length-1].lng.toFixed(6)}}`;
            
            // Fit bounds to trajectory
            const bounds = new google.maps.LatLngBounds();
            trajectoryPoints.forEach(point => bounds.extend(point));
            map.fitBounds(bounds);
        }}
    </script>
    
    <script src="https://maps.googleapis.com/maps/api/js?key={api_key}&callback=initMap" async defer></script>
</body>
</html>"""
        return html
    
    @staticmethod
    def _generate_leaflet_html(points, center_lat, center_lng):
        """
        Generate HTML with Leaflet/OpenStreetMap (free, no API key needed).
        This is the default option and works immediately.
        """
        points_json = json.dumps(points)
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Drone Trajectory - OpenStreetMap</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    <!-- Leaflet CSS -->
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: Arial, sans-serif;
        }}
        #map {{
            height: 100vh;
            width: 100%;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }}
        .info-panel h3 {{
            margin: 0 0 10px 0;
            color: #2563eb;
        }}
        .info-panel p {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .legend {{
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="info-panel">
        <h3>🛩️ Drone Trajectory</h3>
        <p><strong>Total Points:</strong> <span id="point-count">{len(points)}</span></p>
        <p><strong>Start:</strong> <span id="start-coords"></span></p>
        <p><strong>End:</strong> <span id="end-coords"></span></p>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2563eb; height: 3px;"></div>
                <span>Flight Path</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981; width: 12px; height: 12px; border-radius: 50%;"></div>
                <span>Start Point</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444; width: 12px; height: 12px; border-radius: 50%;"></div>
                <span>End Point</span>
            </div>
        </div>
    </div>
    
    <div id="map"></div>
    
    <!-- Leaflet JS -->
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    
    <script>
        const trajectoryPoints = {points_json};
        
        // Initialize map
        const map = L.map('map').setView([{center_lat}, {center_lng}], 16);
        
        // Add tile layers (user can switch between them)
        const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '© OpenStreetMap contributors',
            maxZoom: 19
        }});
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
            attribution: 'Tiles © Esri',
            maxZoom: 19
        }});
        
        // Add default layer
        satelliteLayer.addTo(map);
        
        // Add layer control
        const baseMaps = {{
            "Satellite": satelliteLayer,
            "Street Map": osmLayer
        }};
        L.control.layers(baseMaps).addTo(map);
        
        // Convert points to Leaflet LatLng format
        const latLngs = trajectoryPoints.map(p => [p.lat, p.lng]);
        
        // Draw the flight path
        const polyline = L.polyline(latLngs, {{
            color: '#2563eb',
            weight: 3,
            opacity: 0.8,
            smoothFactor: 1
        }}).addTo(map);
        
        // Add start marker (green)
        const startMarker = L.circleMarker([trajectoryPoints[0].lat, trajectoryPoints[0].lng], {{
            radius: 8,
            fillColor: '#10b981',
            fillOpacity: 1,
            color: 'white',
            weight: 2
        }}).addTo(map);
        startMarker.bindPopup('<b>Start Point</b><br>' + 
            trajectoryPoints[0].lat.toFixed(6) + ', ' + trajectoryPoints[0].lng.toFixed(6));
        
        // Add end marker (red)
        const endMarker = L.circleMarker([trajectoryPoints[trajectoryPoints.length-1].lat, 
                                          trajectoryPoints[trajectoryPoints.length-1].lng], {{
            radius: 8,
            fillColor: '#ef4444',
            fillOpacity: 1,
            color: 'white',
            weight: 2
        }}).addTo(map);
        endMarker.bindPopup('<b>End Point</b><br>' + 
            trajectoryPoints[trajectoryPoints.length-1].lat.toFixed(6) + ', ' + 
            trajectoryPoints[trajectoryPoints.length-1].lng.toFixed(6));
        
        // Add waypoint markers (every 10th point)
        for (let i = 10; i < trajectoryPoints.length - 1; i += 10) {{
            L.circleMarker([trajectoryPoints[i].lat, trajectoryPoints[i].lng], {{
                radius: 4,
                fillColor: '#f59e0b',
                fillOpacity: 0.6,
                color: 'white',
                weight: 1
            }}).addTo(map)
              .bindPopup(`Waypoint ${{i}}<br>${{trajectoryPoints[i].lat.toFixed(6)}}, ${{trajectoryPoints[i].lng.toFixed(6)}}`);
        }}
        
        // Update info panel
        document.getElementById('start-coords').textContent = 
            `${{trajectoryPoints[0].lat.toFixed(6)}}, ${{trajectoryPoints[0].lng.toFixed(6)}}`;
        document.getElementById('end-coords').textContent = 
            `${{trajectoryPoints[trajectoryPoints.length-1].lat.toFixed(6)}}, ${{trajectoryPoints[trajectoryPoints.length-1].lng.toFixed(6)}}`;
        
        // Fit bounds to trajectory
        map.fitBounds(polyline.getBounds(), {{ padding: [50, 50] }});
    </script>
</body>
</html>"""
        return html
    
    @staticmethod
    def create_kml_file(centers, zoom, save_dir):
        """
        Create a KML file for Google Earth visualization.
        
        Args:
            centers: List of (tile_x, tile_y) coordinates
            zoom: Zoom level used for tiles
            save_dir: Directory to save KML file
        """
        if not centers or len(centers) < 2:
            logger.warning("✗ Insufficient trajectory points for KML generation")
            return
        
        # Convert all tile coordinates to lat/lon
        latlon_points = []
        for tile_x, tile_y in centers:
            lat, lon = GoogleMapsTrajectoryPlotter.tile_to_latlon(tile_x, tile_y, zoom)
            latlon_points.append((lon, lat))  # KML uses lon,lat order
        
        # Generate KML content
        kml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>Drone Trajectory</name>
    <description>Flight path with {len(latlon_points)} waypoints</description>
    
    <Style id="flightPath">
      <LineStyle>
        <color>ff0000ff</color>
        <width>3</width>
      </LineStyle>
    </Style>
    
    <Style id="startPoint">
      <IconStyle>
        <color>ff00ff00</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/grn-circle.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <Style id="endPoint">
      <IconStyle>
        <color>ff0000ff</color>
        <scale>1.2</scale>
        <Icon>
          <href>http://maps.google.com/mapfiles/kml/paddle/red-circle.png</href>
        </Icon>
      </IconStyle>
    </Style>
    
    <Placemark>
      <name>Start</name>
      <styleUrl>#startPoint</styleUrl>
      <Point>
        <coordinates>{latlon_points[0][0]},{latlon_points[0][1]},0</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>End</name>
      <styleUrl>#endPoint</styleUrl>
      <Point>
        <coordinates>{latlon_points[-1][0]},{latlon_points[-1][1]},0</coordinates>
      </Point>
    </Placemark>
    
    <Placemark>
      <name>Flight Path</name>
      <styleUrl>#flightPath</styleUrl>
      <LineString>
        <tessellate>1</tessellate>
        <coordinates>
"""
        
        # Add all coordinates
        for lon, lat in latlon_points:
            kml_content += f"          {lon},{lat},0\n"
        
        kml_content += """        </coordinates>
      </LineString>
    </Placemark>
  </Document>
</kml>"""
        
        # Save KML file
        kml_path = os.path.join(save_dir, "trajectory.kml")
        with open(kml_path, 'w', encoding='utf-8') as f:
            f.write(kml_content)
        
        logger.info(f"✓ KML file saved: {kml_path}")
        logger.info(f"  Open in Google Earth for 3D visualization")


# Add this function to your VideoProcessor class or call it in main()
def add_google_maps_visualization(centers, zoom, save_dir, api_key=None):
    """
    Add this call after trajectory processing completes.
    
    Args:
        centers: List of (tile_x, tile_y) trajectory points
        zoom: Zoom level
        save_dir: Results directory
        api_key: Optional Google Maps API key (None = use free OpenStreetMap)
    """
    # Create interactive HTML map
    GoogleMapsTrajectoryPlotter.create_google_maps_html(
        centers, zoom, save_dir, api_key=api_key
    )
    
    # Create KML for Google Earth
    GoogleMapsTrajectoryPlotter.create_kml_file(
        centers, zoom, save_dir
    )