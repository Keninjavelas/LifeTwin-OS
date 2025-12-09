import React from "react";
import { NavigationContainer } from "@react-navigation/native";
import { createNativeStackNavigator } from "@react-navigation/native-stack";
import HomeDashboardScreen from "@screens/HomeDashboardScreen";
import AppUsageSummaryScreen from "@screens/AppUsageSummaryScreen";
import InsightsScreen from "@screens/InsightsScreen";
import SettingsScreen from "@screens/SettingsScreen";
import PermissionsOnboardingScreen from "@screens/PermissionsOnboardingScreen";

export type RootStackParamList = {
  PermissionsOnboarding: undefined;
  Home: undefined;
  AppUsageSummary: undefined;
  Insights: undefined;
  Settings: undefined;
  ModelDebug: undefined;
  Automation: undefined;
  KeystoreDebug: undefined;
};

const Stack = createNativeStackNavigator<RootStackParamList>();

const App = () => {
  const hasPermissions = true; // TODO: wire to native permissions status

  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName={hasPermissions ? "Home" : "PermissionsOnboarding"}>
        <Stack.Screen name="PermissionsOnboarding" component={PermissionsOnboardingScreen} />
        <Stack.Screen name="Home" component={HomeDashboardScreen} />
        <Stack.Screen name="AppUsageSummary" component={AppUsageSummaryScreen} />
        <Stack.Screen name="Insights" component={InsightsScreen} />
        <Stack.Screen name="Settings" component={SettingsScreen} />
        <Stack.Screen name="ModelDebug" component={require('@screens/ModelDebugScreen').default} />
        <Stack.Screen name="Automation" component={require('@screens/AutomationScreen').default} />
        <Stack.Screen name="KeystoreDebug" component={require('@screens/KeystoreDebugScreen').default} />
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default App;
