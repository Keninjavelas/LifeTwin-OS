import React from 'react';
import Head from 'next/head';
import SimulationDashboard from '../components/SimulationDashboard';

const SimulationPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Behavioral Simulation - LifeTwin OS</title>
        <meta name="description" content="Explore how lifestyle changes affect your digital wellbeing" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <main>
        <SimulationDashboard />
      </main>

      <style jsx global>{`
        * {
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }

        body {
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 
                       'Helvetica Neue', Arial, sans-serif;
          background-color: #f5f5f5;
          color: #333;
          line-height: 1.6;
        }

        main {
          min-height: 100vh;
          padding: 20px 0;
        }
      `}</style>
    </>
  );
};

export default SimulationPage;