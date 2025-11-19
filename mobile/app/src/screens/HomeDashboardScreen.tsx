import React from "react";
import { View, Text, Button } from "react-native";
import { NativeStackScreenProps } from "@react-navigation/native-stack";
import { RootStackParamList } from "../App";
import { useAppStore } from "@state/store";

export type HomeDashboardProps = NativeStackScreenProps<RootStackParamList, "Home">;

const HomeDashboardScreen: React.FC<HomeDashboardProps> = ({ navigation }) => {
  const dailySummary = useAppStore((s) => s.dailySummary);
  const predictNextApp = useAppStore((s) => s.predictNextApp);

  const nextApp = predictNextApp();

  return (
    <View style={{ flex: 1, padding: 16 }}>
      <Text style={{ fontSize: 24, fontWeight: "bold", marginBottom: 16 }}>LifeTwin OS — Today</Text>
      <Text>Total screen time: {dailySummary?.totalScreenTime ?? 0} min</Text>
      <Text>Most used apps: {dailySummary?.topApps.join(", ") ?? "-"}</Text>
      <Text>Notifications: {dailySummary?.notificationCount ?? 0}</Text>
      <Text>Predicted next app: {nextApp ?? "(insufficient data)"}</Text>

      <View style={{ marginTop: 24 }}>
        <Button title="App Usage Summary" onPress={() => navigation.navigate("AppUsageSummary")} />
      </View>
      <View style={{ marginTop: 8 }}>
        <Button title="Insights" onPress={() => navigation.navigate("Insights")} />
      </View>
      <View style={{ marginTop: 8 }}>
        <Button title="Settings" onPress={() => navigation.navigate("Settings")} />
      </View>
    </View>
  );
};

export default HomeDashboardScreen;
