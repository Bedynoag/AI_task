import streamlit as st
import openrouteservice
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import folium
import pandas as pd
from streamlit_folium import st_folium

# ====== CONFIGURATION ======
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6ImJiNmZjNzk2YzA0NTQwMzJhY2Q5ZmYzMTBmN2M5MDEzIiwiaCI6Im11cm11cjY0In0="

depot = {
    "name": "Distribution Center - Dallas, TX",
    "coords": [-96.7969, 32.7767]
}

# Store mapping (store code → location & coords)
store_code_to_location = {
    "CA_1": {"name": "Sacramento-Roseville, CA", "coords": [-121.4944, 38.5816]},
    "CA_2": {"name": "San Jose-San Francisco-Oakland, CA", "coords": [-122.4194, 37.7749]},
    "CA_3": {"name": "San Diego-Carlsbad, CA", "coords": [-117.1611, 32.7157]},
    "CA_4": {"name": "Los Angeles, CA", "coords": [-118.2437, 34.0522]},
    "TX_1": {"name": "Corpus Christi-Kingsville-Alice, TX", "coords": [-97.3964, 27.8006]},
    "TX_2": {"name": "El Paso-Las Cruces, TX-NM", "coords": [-106.4850, 31.7619]},
    "TX_3": {"name": "Houston-The Woodlands, TX", "coords": [-95.3698, 29.7604]},
    "WI_1": {"name": "Milwaukee-Racine-Waukesha, WI", "coords": [-87.9065, 43.0389]},
    "WI_2": {"name": "Chicago-Naperville-Elgin, IL", "coords": [-87.6298, 41.8781]},
    "WI_3": {"name": "Madison, WI", "coords": [-89.4012, 43.0731]},
}

def get_route_geometry(client, start_coord, end_coord):
    """Fetch road geometry between two coordinates using truck profile."""
    try:
        route = client.directions(
            coordinates=[start_coord, end_coord],
            profile='driving-hgv',  # Truck profile for real scenario
            format='geojson'
        )
        geometry = route['features'][0]['geometry']['coordinates']
        return [(lat, lon) for lon, lat in geometry]
    except Exception as e:
        print(f"Failed to get route: {e}")
        return None

@st.cache_data
def run_vrp_from_forecast(parsed_query, forecast_data_dict, vehicle_capacity=150):
    """
    Runs VRP using forecast totals as demands, optimizing for time and distance in real scenario.
    parsed_query: dict from chatbot.parse_query() → must contain 'stores'
    forecast_data_dict: dict {store_code: DataFrame} from get_forecast_data()
    """
    stores_selected = parsed_query["stores"]

    # Calculate total demand for each store from forecast data
    demands_selected = []
    for store in stores_selected:
        if store in forecast_data_dict:
            df = forecast_data_dict[store]
            # Sum all numeric columns except 'id'
            total_demand = df.drop(columns=["id"], errors="ignore").apply(pd.to_numeric, errors="coerce").sum(axis=1).iloc[0]
            demands_selected.append(int(total_demand))
        else:
            demands_selected.append(0)  # no data → zero demand

    # Build selected store list
    stores = [store_code_to_location[code] for code in stores_selected]

    # Init ORS client
    client = openrouteservice.Client(key=API_KEY)

    # Combine depot + stores
    all_locations = [depot] + stores
    all_coordinates = [loc["coords"] for loc in all_locations]
    all_demands = [0] + demands_selected
    DEPOT_INDEX = 0

    # Get matrix with both distance and duration for real-road calculation
    try:
        matrix_response = client.distance_matrix(
            locations=all_coordinates,
            profile='driving-hgv',  # Truck profile for real scenario
            metrics=['distance', 'duration'],
            units='km'
        )
        distance_matrix = matrix_response['distances']
        duration_matrix = matrix_response['durations']  # In seconds
    except Exception as e:
        st.error(f"Error fetching matrix: {str(e)}")
        return None

    # Setup OR-Tools routing, optimize for time (duration)
    num_vehicles = 1
    manager = pywrapcp.RoutingIndexManager(len(all_locations), num_vehicles, DEPOT_INDEX)
    routing = pywrapcp.RoutingModel(manager)

    def duration_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(duration_matrix[from_node][to_node])  # Use duration as cost to minimize time

    transit_callback_index = routing.RegisterTransitCallback(duration_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return all_demands[manager.IndexToNode(from_index)]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        [vehicle_capacity],
        True,
        'Capacity'
    )

    # Skip zero-demand stores
    for i in range(len(all_locations)):
        if i != DEPOT_INDEX and all_demands[i] == 0:
            routing.AddDisjunction([manager.NodeToIndex(i)], 1)

    # Solve VRP with time limit and cheapest arc strategy for optimal route
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(30)  # Increased time for better optimization
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("No solution found!")
        return None

    # Extract route order and calculate totals
    index = routing.Start(0)
    route_order = []
    total_distance = 0.0
    total_time = 0.0  # In seconds
    while not routing.IsEnd(index):
        node = manager.IndexToNode(index)
        route_order.append(node)
        next_index = solution.Value(routing.NextVar(index))
        if next_index != routing.End(0):
            next_node = manager.IndexToNode(next_index)
            total_distance += distance_matrix[node][next_node]
            total_time += duration_matrix[node][next_node]
        index = next_index

    # Calculate cost (real scenario: fuel + driver time)
    fuel_cost_per_km = 0.5  # Example: $0.5 per km
    driver_cost_per_hour = 20  # Example: $20 per hour
    total_cost = (total_distance * fuel_cost_per_km) + ((total_time / 3600) * driver_cost_per_hour)

    # Build route map
    route_coords = [all_locations[node]["coords"] for node in route_order] + [depot["coords"]]
    m = folium.Map(location=[depot["coords"][1], depot["coords"][0]], zoom_start=4)

    # Depot marker
    folium.Marker(
        location=[depot["coords"][1], depot["coords"][0]],
        popup=f"🏭 DEPOT<br>{depot['name']}",
        tooltip="Depot",
        icon=folium.Icon(color='red', icon='home')
    ).add_to(m)

    # Store markers with stop order
    stop_counter = 1
    route_names = [depot['name']]  # For summary
    for node in route_order[1:]:  # Skip depot
        loc = all_locations[node]
        demand = all_demands[node]
        folium.Marker(
            location=[loc["coords"][1], loc["coords"][0]],
            popup=f"📦 Stop #{stop_counter}<br>{loc['name']}<br>Demand: {demand}",
            tooltip=f"Stop #{stop_counter}",
            icon=folium.DivIcon(html=f'<div style="background-color: #4CAF50; color: white; border-radius: 50%; width: 30px; height: 30px; text-align: center; line-height: 30px; font-weight: bold;">{stop_counter}</div>')
        ).add_to(m)
        route_names.append(loc['name'])
        stop_counter += 1

    # Draw route lines
    for i in range(len(route_coords) - 1):
        road_line = get_route_geometry(client, route_coords[i], route_coords[i + 1])
        if road_line:
            folium.PolyLine(road_line, color="green", weight=4, opacity=0.8).add_to(m)

    # Add route summary to map
    route_summary = f"Route Order: {' -> '.join(route_names)} -> {depot['name']}<br>Total Distance: {total_distance:.2f} km<br>Total Time: {total_time / 3600:.2f} hours<br>Total Cost: ${total_cost:.2f}"
    folium.Marker(
        location=[depot["coords"][1] + 0.1, depot["coords"][0]],
        popup=route_summary,
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(m)

    return m, route_order, total_distance, total_time, total_cost